#!/usr/bin/env python3
"""
build_meca_pose_from_axis_z.py
-----------------------------------

Second step of the pipeline.

Input from the vision pipeline:
    - p_pre_mm / p_approach_cam_mm:
        Approach point in camera frame, in millimetres.
    - z_axis_cam / v_in_cam / v_trocar:
        Insertion direction in camera frame.

This script:
    1. Loads the camera-to-robot calibration T_base_cam.
    2. Transforms p_pre_cam and z_axis_cam to meca_base_link.
    3. Builds the TCP orientation using your supervisor's method:

        default_orientation = RPY(-180, 0, 90)
        src = default_rotation @ tool_axis_local
        dst = z_axis_base

        q_align = minimal rotation from src to dst
        approach_orientation = q_align * default_orientation

    4. Saves T_base_target, T_base_cam and prints the Meca500 MovePose values.

Notes:
    - The calibration YAML is interpreted as T_base_cam when:
        frame_id       = meca_base_link
        child_frame_id = zivid/pcl/camera frame
    - The q_align logic handles the 3 cases:
        src == dst, src == -dst, and the general case.
    - Quaternions are stored in xyzw order, like ROS/tf_transformations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import tf_transformations as tft
    HAS_TFT = True
except Exception:
    tft = None
    HAS_TFT = False

try:
    from scipy.spatial.transform import Rotation as SciRot
    HAS_SCIPY = True
except Exception:
    SciRot = None
    HAS_SCIPY = False


CALIB_PATHS = (
    Path("calibration/tf2_zivid_robotBase.generated.yaml"),
    Path("tf2_zivid_robotBase.generated.yaml"),
)


# ============================================================
# Basic utilities
# ============================================================

def normalize(v: Any, name: str = "vector") -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))

    if n < 1e-9:
        raise ValueError(f"Cannot normalize near-zero {name}: {v}")

    return v / n


def as_vec3(value: Any, name: str) -> np.ndarray:
    """
    Convert common JSON representations to a 3-vector.

    Accepted:
        [x, y, z]
        {"x": x, "y": y, "z": z}
        {"0": x, "1": y, "2": z}
    """
    if isinstance(value, dict):
        if all(k in value for k in ("x", "y", "z")):
            value = [value["x"], value["y"], value["z"]]
        elif all(k in value for k in ("0", "1", "2")):
            value = [value["0"], value["1"], value["2"]]

    arr = np.asarray(value, dtype=float).reshape(-1)

    if arr.size != 3:
        raise ValueError(f"{name} must contain exactly 3 values. Got: {value}")

    return arr.astype(float)


def find_first_key(data: dict, keys: tuple[str, ...], name: str) -> np.ndarray:
    for key in keys:
        if key in data and data[key] is not None:
            return as_vec3(data[key], key)

    raise KeyError(
        f"Could not find {name}. Tried keys: {', '.join(keys)}"
    )


# ============================================================
# Minimal tf_transformations wrappers
# ============================================================

def _axes_to_scipy_seq(axes: str) -> str:
    """
    Minimal mapping for the axes used here.

    tf_transformations:
        sxyz = static/extrinsic XYZ
        rxyz = rotating/intrinsic XYZ

    scipy:
        xyz = extrinsic XYZ
        XYZ = intrinsic XYZ
    """
    axes = axes.lower()

    if axes == "sxyz":
        return "xyz"
    if axes == "rxyz":
        return "XYZ"

    raise ValueError(
        f"Fallback without tf_transformations only supports axes='sxyz' or 'rxyz'. Got: {axes}"
    )


def quaternion_from_euler(roll_rad: float, pitch_rad: float, yaw_rad: float, axes: str = "sxyz") -> np.ndarray:
    """
    Return quaternion in xyzw order.
    """
    if HAS_TFT:
        return np.asarray(
            tft.quaternion_from_euler(roll_rad, pitch_rad, yaw_rad, axes=axes),
            dtype=float,
        )

    if not HAS_SCIPY:
        raise ImportError("Need either tf_transformations or scipy installed.")

    seq = _axes_to_scipy_seq(axes)
    return SciRot.from_euler(seq, [roll_rad, pitch_rad, yaw_rad], degrees=False).as_quat()


def quaternion_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """
    Return 4x4 homogeneous rotation matrix from xyzw quaternion.
    """
    q_xyzw = np.asarray(q_xyzw, dtype=float).reshape(4)

    if HAS_TFT:
        return np.asarray(tft.quaternion_matrix(q_xyzw), dtype=float)

    if not HAS_SCIPY:
        raise ImportError("Need either tf_transformations or scipy installed.")

    M = np.eye(4)
    M[:3, :3] = SciRot.from_quat(q_xyzw).as_matrix()
    return M


def quaternion_multiply(q1_xyzw: np.ndarray, q2_xyzw: np.ndarray) -> np.ndarray:
    """
    Return q = q1 * q2 in xyzw order.
    Equivalent to R(q) = R(q1) @ R(q2).
    """
    q1_xyzw = np.asarray(q1_xyzw, dtype=float).reshape(4)
    q2_xyzw = np.asarray(q2_xyzw, dtype=float).reshape(4)

    if HAS_TFT:
        return np.asarray(tft.quaternion_multiply(q1_xyzw, q2_xyzw), dtype=float)

    # Manual Hamilton product for xyzw.
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=float)


def euler_from_matrix(M: np.ndarray, axes: str = "rxyz") -> tuple[float, float, float]:
    """
    Return Euler angles in radians.

    For Meca500 MovePose, use axes='rxyz' for mobile/intrinsic XYZ.
    """
    if HAS_TFT:
        return tuple(float(v) for v in tft.euler_from_matrix(M, axes=axes))

    if not HAS_SCIPY:
        raise ImportError("Need either tf_transformations or scipy installed.")

    seq = _axes_to_scipy_seq(axes)
    Rm = np.asarray(M, dtype=float)[:3, :3]
    return tuple(float(v) for v in SciRot.from_matrix(Rm).as_euler(seq, degrees=False))


# ============================================================
# Calibration loading
# ============================================================

def find_calibration_path(user_path: Path | None = None) -> Path:
    if user_path is not None:
        if not user_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {user_path}")
        return user_path

    for path in CALIB_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Calibration file not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in CALIB_PATHS)
    )


def _find_ros_parameters(node: Any) -> dict:
    """
    Recursively find the dictionary containing ROS static transform parameters.
    """
    if isinstance(node, dict):
        if "ros__parameters" in node:
            return node["ros__parameters"]

        if all(k in node for k in ("translation", "rotation")):
            return node

        for value in node.values():
            try:
                return _find_ros_parameters(value)
            except KeyError:
                pass

    raise KeyError("Could not find ros__parameters with translation and rotation.")


def load_T_base_cam(
    calib_path: Path | None = None,
    translation_units: str = "auto",
    invert: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Load T_base_cam from a ROS-generated YAML.

    Expected YAML meaning:
        parent/frame_id = meca_base_link
        child_frame_id = camera/zivid/pcl frame

    If your YAML is the opposite direction, use --invert-calib.
    """
    path = find_calibration_path(calib_path)

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    params = _find_ros_parameters(data)

    trans = params["translation"]
    rot = params["rotation"]

    t = np.array(
        [float(trans["x"]), float(trans["y"]), float(trans["z"])],
        dtype=float,
    )

    if translation_units == "auto":
        # If values are small, they are probably metres. If they are hundreds,
        # they are already millimetres.
        units_used = "m" if np.max(np.abs(t)) < 10.0 else "mm"
    else:
        units_used = translation_units

    if units_used == "m":
        t = 1000.0 * t
    elif units_used != "mm":
        raise ValueError("translation_units must be 'auto', 'mm', or 'm'.")

    q_xyzw = np.array(
        [float(rot["x"]), float(rot["y"]), float(rot["z"]), float(rot["w"])],
        dtype=float,
    )

    R_base_cam = quaternion_matrix(q_xyzw)[:3, :3]

    T = np.eye(4)
    T[:3, :3] = R_base_cam
    T[:3, 3] = t

    if invert:
        T = np.linalg.inv(T)

    info = {
        "calib_path": str(path),
        "frame_id": params.get("frame_id"),
        "child_frame_id": params.get("child_frame_id"),
        "interpreted_as": "T_base_cam" if not invert else "inverse(original_yaml)",
        "translation_units_used": units_used,
        "translation_mm": t.tolist(),
        "quaternion_xyzw": q_xyzw.tolist(),
        "inverted": bool(invert),
    }

    return T, info


# ============================================================
# Supervisor method: q_align * default_orientation
# ============================================================

def q_align_supervisor_method(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Exact supervisor logic for the minimal alignment quaternion.

    Cases:
        1. src ≈ dst:
            q_align = identity

        2. src ≈ -dst:
            cross(src, dst) is zero, so choose a perpendicular fallback axis
            and rotate 180 degrees.

        3. General case:
            rotation_axis  = cross(src, dst)
            rotation_angle = acos(dot(src, dst))
            q_align        = quaternion(axis, angle)

    Returns:
        q_align_xyzw, debug_info
    """
    src = normalize(src, "src")
    dst = normalize(dst, "dst")

    dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))

    if dot > 1.0 - 1e-9:
        q_align = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        case = "already_aligned"
        rotation_axis = np.array([0.0, 0.0, 0.0], dtype=float)
        rotation_angle = 0.0

    elif dot < -1.0 + 1e-9:
        # Singular antiparallel case: cross(src, dst) is zero.
        axis = np.cross(src, np.array([1.0, 0.0, 0.0]))

        if np.linalg.norm(axis) < 1e-9:
            axis = np.cross(src, np.array([0.0, 1.0, 0.0]))

        axis = axis / np.linalg.norm(axis)
        rotation_axis = axis
        rotation_angle = math.pi

        q_align = np.array([
            axis[0] * math.sin(math.pi / 2.0),
            axis[1] * math.sin(math.pi / 2.0),
            axis[2] * math.sin(math.pi / 2.0),
            math.cos(math.pi / 2.0),
        ], dtype=float)

        case = "antiparallel_180deg_fallback"

    else:
        rotation_axis = np.cross(src, dst)
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        rotation_angle = math.acos(dot)

        q_align = np.array([
            rotation_axis[0] * math.sin(rotation_angle / 2.0),
            rotation_axis[1] * math.sin(rotation_angle / 2.0),
            rotation_axis[2] * math.sin(rotation_angle / 2.0),
            math.cos(rotation_angle / 2.0),
        ], dtype=float)

        case = "general_cross_acos"

    q_align = q_align / np.linalg.norm(q_align)

    info = {
        "case": case,
        "src": src.tolist(),
        "dst": dst.tolist(),
        "dot_src_dst": dot,
        "rotation_axis": rotation_axis.tolist(),
        "rotation_angle_rad": float(rotation_angle),
        "rotation_angle_deg": float(math.degrees(rotation_angle)),
        "q_align_xyzw": q_align.tolist(),
    }

    return q_align, info


def build_approach_orientation_like_supervisor(
    z_axis_base: np.ndarray,
    default_euler_deg: tuple[float, float, float] = (-180.0, 0.0, 90.0),
    default_euler_axes: str = "sxyz",
    tool_axis_local: np.ndarray = np.array([0.0, 0.0, 1.0]),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Implements exactly:

        default_orientation = RPY(-180, 0, 90)
        default_rotation = quaternion_matrix(default_orientation)[:3, :3]
        tool_axis_local = [0, 0, 1]

        src = default_rotation @ tool_axis_local
        dst = z_axis_base

        q_align = minimal rotation src -> dst
        approach_orientation = q_align * default_orientation

    Returns:
        approach_orientation_xyzw, R_base_tcp, debug_info
    """
    dst = normalize(z_axis_base, "z_axis_base")
    tool_axis_local = normalize(tool_axis_local, "tool_axis_local")

    default_orientation = quaternion_from_euler(
        math.radians(default_euler_deg[0]),
        math.radians(default_euler_deg[1]),
        math.radians(default_euler_deg[2]),
        axes=default_euler_axes,
    )
    default_orientation = default_orientation / np.linalg.norm(default_orientation)

    default_rotation = quaternion_matrix(default_orientation)[:3, :3]

    # src = where the local TCP tool axis points under the default orientation.
    src = default_rotation @ tool_axis_local
    src = normalize(src, "src")

    q_align, align_info = q_align_supervisor_method(src=src, dst=dst)

    # approach_orientation = q_align * default_orientation
    approach_orientation = quaternion_multiply(q_align, default_orientation)
    approach_orientation = approach_orientation / np.linalg.norm(approach_orientation)

    R_base_tcp = quaternion_matrix(approach_orientation)[:3, :3]

    final_axis = normalize(R_base_tcp @ tool_axis_local, "final_axis")
    alignment = float(np.dot(final_axis, dst))

    if alignment < 0.99:
        raise RuntimeError(
            f"Bad TCP axis alignment. final_axis·dst = {alignment:.6f}. "
            "If this is approximately -1, try --tool-axis-local 0 0 -1."
        )

    info = {
        "method": "supervisor_q_align_times_default_orientation",
        "tf_transformations_available": HAS_TFT,
        "scipy_available": HAS_SCIPY,
        "default_euler_deg": list(default_euler_deg),
        "default_euler_axes": default_euler_axes,
        "default_orientation_xyzw": default_orientation.tolist(),
        "tool_axis_local": tool_axis_local.tolist(),
        "src_default_axis_base": src.tolist(),
        "dst_z_axis_base": dst.tolist(),
        "q_align_info": align_info,
        "approach_orientation_xyzw": approach_orientation.tolist(),
        "final_axis_base": final_axis.tolist(),
        "alignment": alignment,
    }

    return approach_orientation, R_base_tcp, info


# ============================================================
# Prediction loading
# ============================================================

def load_camera_geometry_from_prediction(prediction_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Common structures:
    #   prediction.json contains pose fields at top level
    #   result_summary.json may contain them under "pose"
    if "pose" in data and isinstance(data["pose"], dict):
        pose_data = data["pose"]
    else:
        pose_data = data

    p_pre_cam = find_first_key(
        pose_data,
        keys=(
            "p_pre_mm",
            "p_approach_cam_mm",
            "p_target_mm",
            "p_pre_cam_mm",
            "p_approach_mm",
        ),
        name="approach point in camera frame",
    )

    z_axis_cam = find_first_key(
        pose_data,
        keys=(
            "z_axis_cam",
            "v_in_cam",
            "v_trocar",
            "z_axis",
            "local_plane_normal_cam",
        ),
        name="insertion z-axis in camera frame",
    )

    return p_pre_cam, normalize(z_axis_cam, "z_axis_cam"), pose_data


# ============================================================
# Main conversion
# ============================================================

def convert_camera_axis_to_meca_pose(
    p_pre_cam_mm: np.ndarray,
    z_axis_cam: np.ndarray,
    T_base_cam: np.ndarray,
    default_euler_deg: tuple[float, float, float],
    default_euler_axes: str,
    tool_axis_local: np.ndarray,
    meca_euler_axes: str,
) -> tuple[np.ndarray, dict]:
    R_base_cam = T_base_cam[:3, :3]
    t_base_cam = T_base_cam[:3, 3]

    # Point: rotate + translate.
    p_pre_base_mm = R_base_cam @ np.asarray(p_pre_cam_mm, dtype=float).reshape(3) + t_base_cam

    # Direction vector: rotate only, no translation.
    z_axis_base = normalize(R_base_cam @ normalize(z_axis_cam, "z_axis_cam"), "z_axis_base")

    approach_orientation, R_base_tcp, orientation_info = build_approach_orientation_like_supervisor(
        z_axis_base=z_axis_base,
        default_euler_deg=default_euler_deg,
        default_euler_axes=default_euler_axes,
        tool_axis_local=np.asarray(tool_axis_local, dtype=float),
    )

    T_base_target = np.eye(4)
    T_base_target[:3, :3] = R_base_tcp
    T_base_target[:3, 3] = p_pre_base_mm

    # Euler angles for Meca500 MovePose.
    # By default this uses rxyz, i.e. rotating/mobile/intrinsic XYZ.
    alpha, beta, gamma = euler_from_matrix(T_base_target, axes=meca_euler_axes)
    euler_deg = np.degrees([alpha, beta, gamma])

    result = {
        "p_pre_cam_mm": np.asarray(p_pre_cam_mm, dtype=float).reshape(3).tolist(),
        "z_axis_cam": normalize(z_axis_cam, "z_axis_cam").tolist(),
        "p_pre_base_mm": p_pre_base_mm.tolist(),
        "z_axis_base": z_axis_base.tolist(),
        "R_base_tcp": R_base_tcp.tolist(),
        "T_base_target": T_base_target.tolist(),
        "approach_orientation_xyzw": approach_orientation.tolist(),
        "meca_euler_axes": meca_euler_axes,
        "meca_move_pose": {
            "x_mm": float(p_pre_base_mm[0]),
            "y_mm": float(p_pre_base_mm[1]),
            "z_mm": float(p_pre_base_mm[2]),
            "alpha_deg": float(euler_deg[0]),
            "beta_deg": float(euler_deg[1]),
            "gamma_deg": float(euler_deg[2]),
        },
        "orientation_info": orientation_info,
    }

    return T_base_target, result


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert camera-frame approach point + insertion z-axis into a "
            "Meca500 base-frame 6DoF target pose using the supervisor q_align method."
        )
    )

    parser.add_argument(
        "--prediction",
        type=Path,
        default=None,
        help=(
            "Path to prediction.json/result_summary.json containing p_pre_mm "
            "and z_axis_cam/v_in_cam/v_trocar."
        ),
    )

    parser.add_argument(
        "--p-pre",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Approach point in camera frame, in mm. Alternative to --prediction.",
    )

    parser.add_argument(
        "--z-axis",
        type=float,
        nargs=3,
        default=None,
        metavar=("ZX", "ZY", "ZZ"),
        help="Insertion direction in camera frame. Alternative to --prediction.",
    )

    parser.add_argument(
        "--calib",
        type=Path,
        default=None,
        help=(
            "Calibration YAML path. If omitted, tries calibration/tf2_zivid_robotBase.generated.yaml "
            "and tf2_zivid_robotBase.generated.yaml."
        ),
    )

    parser.add_argument(
        "--calib-translation-units",
        choices=("auto", "mm", "m"),
        default="auto",
        help="Units of the calibration translation. Default: auto.",
    )

    parser.add_argument(
        "--invert-calib",
        action="store_true",
        help="Invert the calibration transform before using it.",
    )

    parser.add_argument(
        "--default-euler",
        type=float,
        nargs=3,
        default=(-180.0, 0.0, 90.0),
        metavar=("ROLL", "PITCH", "YAW"),
        help="Default orientation used by the supervisor method, in degrees.",
    )

    parser.add_argument(
        "--default-euler-axes",
        type=str,
        default="sxyz",
        help=(
            "Euler axes used to build default_orientation. Default is 'sxyz' to reproduce "
            "tf_transformations.quaternion_from_euler(...) exactly as in the supervisor snippet."
        ),
    )

    parser.add_argument(
        "--meca-euler-axes",
        type=str,
        default="rxyz",
        help=(
            "Euler axes used to convert the final matrix to MovePose angles. "
            "Default 'rxyz' = rotating/mobile/intrinsic XYZ."
        ),
    )

    parser.add_argument(
        "--tool-axis-local",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help=(
            "Local TCP axis that points along the needle. "
            "Use 0 0 1 for +Z_TCP, or 0 0 -1 for -Z_TCP."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: prediction file directory if --prediction is used, "
            "otherwise current directory."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.prediction is not None:
        p_pre_cam_mm, z_axis_cam, _ = load_camera_geometry_from_prediction(args.prediction)
        default_out_dir = args.prediction.parent
    else:
        if args.p_pre is None or args.z_axis is None:
            raise SystemExit(
                "Use either --prediction PATH or both --p-pre X Y Z and --z-axis ZX ZY ZZ."
            )

        p_pre_cam_mm = np.asarray(args.p_pre, dtype=float)
        z_axis_cam = normalize(np.asarray(args.z_axis, dtype=float), "z_axis_cam")
        default_out_dir = Path(".")

    out_dir = args.out_dir if args.out_dir is not None else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    T_base_cam, calib_info = load_T_base_cam(
        calib_path=args.calib,
        translation_units=args.calib_translation_units,
        invert=args.invert_calib,
    )

    T_base_target, result = convert_camera_axis_to_meca_pose(
        p_pre_cam_mm=p_pre_cam_mm,
        z_axis_cam=z_axis_cam,
        T_base_cam=T_base_cam,
        default_euler_deg=tuple(args.default_euler),
        default_euler_axes=args.default_euler_axes,
        tool_axis_local=np.asarray(args.tool_axis_local, dtype=float),
        meca_euler_axes=args.meca_euler_axes,
    )

    result["calibration"] = calib_info

    # Save the calibrated camera -> robot/base transform as well.
    # This is used by the inference pipeline to draw camera-frame geometry
    # such as the fitted eye sphere and trocar/insertion axis in RViz.
    result["T_base_cam"] = T_base_cam.tolist()
    result["T_cam_base"] = np.linalg.inv(T_base_cam).tolist()

    np.save(out_dir / "T_base_cam.npy", T_base_cam)
    np.save(out_dir / "T_cam_base.npy", np.linalg.inv(T_base_cam))

    np.save(out_dir / "T_base_target.npy", T_base_target)
    np.save(out_dir / "R_base_tcp.npy", np.asarray(result["R_base_tcp"]))
    np.save(out_dir / "p_pre_base_mm.npy", np.asarray(result["p_pre_base_mm"]))
    np.save(out_dir / "z_axis_base.npy", np.asarray(result["z_axis_base"]))
    np.save(out_dir / "approach_orientation_xyzw.npy", np.asarray(result["approach_orientation_xyzw"]))

    json_path = out_dir / "meca_pose_from_axis_z.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    mp = result["meca_move_pose"]
    info = result["orientation_info"]
    qinfo = info["q_align_info"]

    print()
    print("Meca500 target pose generated with supervisor q_align method")
    print(f"Calibration: {calib_info['calib_path']}")
    print(f"Output:      {json_path}")
    print(f"Saved:       {out_dir / 'T_base_cam.npy'}")
    print(f"Saved:       {out_dir / 'T_base_target.npy'}")
    print()
    print("Supervisor alignment:")
    print(f"  case:             {qinfo['case']}")
    print(f"  dot(src, dst):    {qinfo['dot_src_dst']:.6f}")
    print(f"  angle:            {qinfo['rotation_angle_deg']:.3f} deg")
    print(f"  axis alignment:   {info['alignment']:.6f}")
    print()
    print("MovePose values:")
    print(
        "MovePose("
        f"{mp['x_mm']:.3f}, "
        f"{mp['y_mm']:.3f}, "
        f"{mp['z_mm']:.3f}, "
        f"{mp['alpha_deg']:.3f}, "
        f"{mp['beta_deg']:.3f}, "
        f"{mp['gamma_deg']:.3f}"
        ")"
    )
    print()
    print("If axis alignment is close to -1, rerun with:")
    print("  --tool-axis-local 0 0 -1")


if __name__ == "__main__":
    main()
