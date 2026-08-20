from pathlib import Path
import json
import subprocess
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import zivid
from ultralytics import YOLO

from crop_utils import crop_rgb_and_xyz, bbox_local_to_global, get_3d_from_mask
from pose_utils_trocarfit import estimate_trocar_pose, filter_points

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float64MultiArray
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except Exception as e:
    rclpy = None
    Node = object
    Float64MultiArray = None
    JointState = None
    ROS2_AVAILABLE = False
    ROS2_IMPORT_ERROR = e


# ============================================================
# Configuration
# ============================================================

SETTINGS_YML = Path("ZIVID/Z2p_M60_Inspection_SmallFeatures_Off.yml")

MODEL_A_PATH = Path("models/model_A_yolo26s.pt")
MODEL_B_PATH = Path("models/model_b_letterbox_best.pt")

OUT_DIR = Path("Inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MECA_POSE_SCRIPT = Path("build_meca_pose_from_axis_z.py")
NATIVE_MOVEPOSE_SCRIPT = Path("/home/lucia/ros2_ws/scripts/send_meca_move_pose_from_target.py")
MOVELIN_ADVANCE_SCRIPT = Path("/home/lucia/ros2_ws/scripts/move_lin_advance_current.py")
J6_ROTATE_SCRIPT = Path("/home/lucia/ros2_ws/scripts/rotate_j6_interactive.py")

MARGIN = 0.12
CONF_A = 0.5
SEG_THRESH = 0.5
IMG_SIZE = 320
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_RETRIES = 3

# Terminal output control.
# Keep False for normal experiments: only essential status messages are printed.
VERBOSE_TERMINAL = False
PRINT_TIMING_BREAKDOWN = False
SHOW_SUBPROCESS_OUTPUT = False

ENABLE_POSE_SAFETY_CHECKS = True
MIN_TROCAR_POINTS_FOR_ROBOT = 90
MIN_EYE_POINTS_FOR_ROBOT = 5000
MIN_SPHERE_INLIERS_FOR_ROBOT = 5000
ENABLE_TEMPORAL_POSE_CHECK = False
MAX_P_PRE_JUMP_MM = 5.0
_last_accepted_p_pre_mm = None

SAFE_POSE_EYE_SIDE = "ask"  # "ask", "right", or "left"
SAFE_JOINTS_LEFT_EYE_DEG = [160.0, 0.0, 0.0, 0.0, 49.0, 0.0]
SAFE_JOINTS_RIGHT_EYE_DEG = [0.0, 0.0, 0.0, 0.0, 49.0, 0.0]
SAFE_PREPOSE_TWO_STEP_BEFORE_J1 = True
SAFE_CURRENT_J1_TIMEOUT_SEC = 2.0
SAFE_JOINT_DISCOVERY_WAIT_SEC = 2.0
SAFE_JOINT_SETTLE_SEC = 0.5
JOINT_TARGET_TOPIC = "/joint_targets"
REQUIRE_ENTER_BEFORE_SAFE_JOINTS = True

AUTO_NATIVE_MOVEPOSE = True
NATIVE_MOVEPOSE_EXTRA_CLEARANCE_MM = 0.0
NATIVE_MOVEPOSE_DRY_RUN = False
REQUIRE_ENTER_BEFORE_NATIVE_MOVEPOSE = True

AUTO_ADVANCE_MOVELIN = True
ADVANCE_MM = 55.0
ADVANCE_AXIS_COLUMN = 2
ADVANCE_AXIS_SIGN = 1.0
REQUIRE_ENTER_BEFORE_MOVELIN_ADVANCE = True

AUTO_J6_ROTATION_STEP = True

AUTO_FINAL_MOVELIN = True
FINAL_MOVELIN_MM = 5.0
REQUIRE_ENTER_BEFORE_FINAL_MOVELIN = True

MECA_JOINT_NAMES = [
    "meca_axis_1_joint",
    "meca_axis_2_joint",
    "meca_axis_3_joint",
    "meca_axis_4_joint",
    "meca_axis_5_joint",
    "meca_axis_6_joint",
]


# ============================================================
# Timing
# ============================================================

def sync_cuda_if_needed():
    if torch.cuda.is_available() and DEVICE == "cuda":
        torch.cuda.synchronize()


def elapsed_ms(t_start):
    return (time.perf_counter() - t_start) * 1000.0


def format_ms(value):
    return "N/A" if value is None else f"{float(value):.1f} ms"


def print_pipeline_timings(timings):
    timings = timings or {}

    if not PRINT_TIMING_BREAKDOWN and not VERBOSE_TERMINAL:
        total = format_ms(timings.get("total_pipeline_time_ms"))
        robot = timings.get("robot_motion_time_ms")
        if robot is None:
            print(f"  Timing: total={total}")
        else:
            print(f"  Timing: total={total}, robot={format_ms(robot)}")
        return

    rows = [
        ("Eye detection", "eye_detection_ms"),
        ("Eye crop generation", "eye_crop_generation_ms"),
        ("Trocar segmentation", "trocar_segmentation_ms"),
        ("Pose estimation", "pose_estimation_ms"),
        ("Robot motion", "robot_motion_time_ms"),
        ("Total pipeline", "total_pipeline_time_ms"),
    ]
    print("  Timing breakdown:")
    for label, key in rows:
        print(f"  {label:<24} {format_ms(timings.get(key))}")


def log_verbose(message: str):
    if VERBOSE_TERMINAL:
        print(message)


def run_command(cmd, *, interactive: bool = False, check: bool = True):
    """Run a subprocess with quiet output by default.

    Successful child-process output is hidden unless SHOW_SUBPROCESS_OUTPUT=True.
    If the command fails, stdout/stderr are printed so the error can be diagnosed.
    Interactive commands are never silenced.
    """
    if interactive or SHOW_SUBPROCESS_OUTPUT:
        return subprocess.run(cmd, check=check)

    completed = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if check and completed.returncode != 0:
        print(f"  Command failed with exit code {completed.returncode}: {' '.join(map(str, cmd))}")
        if completed.stdout and completed.stdout.strip():
            print("  stdout:")
            print(completed.stdout.rstrip())
        if completed.stderr and completed.stderr.strip():
            print("  stderr:")
            print(completed.stderr.rstrip())
        raise subprocess.CalledProcessError(
            completed.returncode,
            cmd,
            output=completed.stdout,
            stderr=completed.stderr,
        )

    return completed


# ============================================================
# Models and preprocessing
# ============================================================

def load_model_b(ckpt_path: Path) -> torch.nn.Module:
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()
    return model


_norm_tf = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def letterbox_rgb(img: np.ndarray, img_size: int = 320):
    orig_h, orig_w = img.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid image shape: {img.shape}")

    scale = img_size / max(orig_h, orig_w)
    new_w = max(1, min(img_size, int(round(orig_w * scale))))
    new_h = max(1, min(img_size, int(round(orig_h * scale))))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = img_size - new_w
    pad_h = img_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_h - pad_top,
        pad_left,
        pad_w - pad_left,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_left": pad_left,
    }
    return padded, meta


def unletterbox_mask(mask_lbox: np.ndarray, meta: dict):
    mask_no_pad = mask_lbox[
        meta["pad_top"]:meta["pad_top"] + meta["new_h"],
        meta["pad_left"]:meta["pad_left"] + meta["new_w"],
    ]
    mask_crop = cv2.resize(
        mask_no_pad.astype(np.uint8),
        (meta["orig_w"], meta["orig_h"]),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask_crop.astype(np.uint8)


def predict_trocar_mask(model_b: torch.nn.Module, rgb_crop: np.ndarray):
    rgb_lbox, meta = letterbox_rgb(rgb_crop, IMG_SIZE)
    inp = _norm_tf(image=rgb_lbox)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model_b(inp)
        prob = torch.sigmoid(logits)
        mask_lbox = (prob > SEG_THRESH).squeeze().cpu().numpy().astype(np.uint8)

    mask_crop = unletterbox_mask(mask_lbox, meta)
    ys, xs = np.where(mask_crop > 0)

    if len(xs) == 0:
        return None, mask_crop

    bbox_local = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return bbox_local, mask_crop


# ============================================================
# Pose safety
# ============================================================

def validate_pose_for_robot(pose):
    global _last_accepted_p_pre_mm

    if pose is None:
        return False, "pose is None"

    num_trocar = int(pose.get("num_trocar_points", 0))
    num_eye = int(pose.get("num_eye_points", 0))
    num_sphere = int(pose.get("num_sphere_inliers", 0))

    if num_trocar < MIN_TROCAR_POINTS_FOR_ROBOT:
        return False, f"too few trocar points: {num_trocar}"
    if num_eye < MIN_EYE_POINTS_FOR_ROBOT:
        return False, f"too few eye points: {num_eye}"
    if num_sphere < MIN_SPHERE_INLIERS_FOR_ROBOT:
        return False, f"too few sphere inliers: {num_sphere}"

    p_pre = pose.get("p_pre_mm")
    p_entry = pose.get("p_entry_trocar_mm")
    z_axis = pose.get("z_axis_cam")

    if p_pre is None:
        return False, "missing p_pre_mm"
    if p_entry is None:
        return False, "missing p_entry_trocar_mm"
    if z_axis is None:
        return False, "missing z_axis_cam"

    p_pre = np.asarray(p_pre, dtype=np.float64).reshape(3)
    p_entry = np.asarray(p_entry, dtype=np.float64).reshape(3)
    z_axis = np.asarray(z_axis, dtype=np.float64).reshape(3)

    if not np.isfinite(p_pre).all():
        return False, "p_pre_mm contains NaN or inf"
    if not np.isfinite(p_entry).all():
        return False, "p_entry_trocar_mm contains NaN or inf"
    if not np.isfinite(z_axis).all():
        return False, "z_axis_cam contains NaN or inf"

    z_norm = float(np.linalg.norm(z_axis))
    if z_norm < 1e-6:
        return False, "z_axis_cam has near-zero norm"
    if abs(z_norm - 1.0) > 0.05:
        return False, f"z_axis_cam is not unit length: norm={z_norm:.3f}"

    expected_clearance = float(pose.get("target_clearance_mm", 50.0))
    measured_clearance = float(np.linalg.norm(p_entry - p_pre))
    if abs(measured_clearance - expected_clearance) > 3.0:
        return False, f"clearance mismatch: {measured_clearance:.2f} mm"

    if ENABLE_TEMPORAL_POSE_CHECK:
        if _last_accepted_p_pre_mm is not None:
            jump = float(np.linalg.norm(p_pre - _last_accepted_p_pre_mm))
            if jump > MAX_P_PRE_JUMP_MM:
                return False, f"p_pre jump too large: {jump:.2f} mm"
        _last_accepted_p_pre_mm = p_pre.copy()

    return True, "pose accepted"


# ============================================================
# Inference
# ============================================================

def run_pipeline(rgb, xyz, model_A, model_B):
    timings_ms = {
        "eye_detection_ms": None,
        "eye_crop_generation_ms": None,
        "trocar_segmentation_ms": None,
        "pose_estimation_ms": None,
        "robot_motion_time_ms": None,
        "total_pipeline_time_ms": None,
    }

    t_step = time.perf_counter()
    sync_cuda_if_needed()
    res_A = model_A(rgb, verbose=False)[0]
    sync_cuda_if_needed()
    timings_ms["eye_detection_ms"] = elapsed_ms(t_step)

    boxes_A = [b for b in res_A.boxes if float(b.conf) >= CONF_A]
    if not boxes_A:
        log_verbose(f"  [A] Eye not detected ({timings_ms['eye_detection_ms']:.1f} ms)")
        return None

    best_A = max(boxes_A, key=lambda b: float(b.conf))
    eye_bbox = best_A.xyxy[0].cpu().numpy().astype(int).tolist()
    conf_A = float(best_A.conf)

    t_step = time.perf_counter()
    rgb_crop, xyz_crop, offset = crop_rgb_and_xyz(rgb, xyz, eye_bbox, margin=MARGIN)
    timings_ms["eye_crop_generation_ms"] = elapsed_ms(t_step)

    if xyz_crop is None:
        return {
            "eye_bbox": eye_bbox,
            "conf_A": round(conf_A, 3),
            "trocar_global": None,
            "rgb_crop": rgb_crop,
            "xyz_crop": None,
            "seg_mask": None,
            "offset": offset,
            "trocar_3d_mm": None,
            "trocar_pointcloud": None,
            "pose": None,
            "timings_ms": timings_ms,
        }

    t_step = time.perf_counter()
    sync_cuda_if_needed()
    trocar_local, seg_mask = predict_trocar_mask(model_B, rgb_crop)
    sync_cuda_if_needed()
    timings_ms["trocar_segmentation_ms"] = elapsed_ms(t_step)

    if trocar_local is None:
        return {
            "eye_bbox": eye_bbox,
            "conf_A": round(conf_A, 3),
            "trocar_global": None,
            "rgb_crop": rgb_crop,
            "xyz_crop": xyz_crop,
            "seg_mask": seg_mask,
            "offset": offset,
            "trocar_3d_mm": None,
            "trocar_pointcloud": None,
            "pose": None,
            "timings_ms": timings_ms,
        }

    trocar_global = bbox_local_to_global(trocar_local, offset)
    trocar_3d, trocar_pointcloud = get_3d_from_mask(xyz_crop, seg_mask)

    pose_result = None
    if trocar_pointcloud is not None:
        t_step = time.perf_counter()
        pose_result = estimate_trocar_pose(xyz_crop, seg_mask, trocar_pointcloud)
        timings_ms["pose_estimation_ms"] = elapsed_ms(t_step)

    return {
        "eye_bbox": eye_bbox,
        "conf_A": round(conf_A, 3),
        "trocar_global": trocar_global,
        "rgb_crop": rgb_crop,
        "xyz_crop": xyz_crop,
        "seg_mask": seg_mask,
        "offset": offset,
        "trocar_3d_mm": trocar_3d.tolist() if trocar_3d is not None else None,
        "trocar_pointcloud": trocar_pointcloud,
        "pose": pose_result,
        "timings_ms": timings_ms,
    }


# ============================================================
# Result files
# ============================================================

def save_rgb_image(path: Path, rgb_img: np.ndarray):
    if rgb_img is not None:
        cv2.imwrite(str(path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))


def save_mask_overlay(rgb_crop, mask_crop, out_path):
    if rgb_crop is None or mask_crop is None:
        return
    overlay = rgb_crop.copy()
    overlay[mask_crop > 0] = [0, 255, 0]
    blended = cv2.addWeighted(rgb_crop, 0.65, overlay, 0.35, 0)
    save_rgb_image(out_path, blended)


def sample_points(points, max_points=5000):
    if points is None:
        return None
    points = np.asarray(points)
    if len(points) <= max_points:
        return points
    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx]


def write_colored_ply(path, points, colors=None):
    points = np.asarray(points, dtype=np.float32)
    if colors is None:
        colors = np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (len(points), 1))
    colors = np.asarray(colors, dtype=np.uint8)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_rgb_xyz_ply(path, xyz, rgb):
    if xyz is None or rgb is None:
        return
    xyz_flat = xyz.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)
    valid = np.isfinite(xyz_flat).all(axis=1)
    valid &= np.linalg.norm(xyz_flat, axis=1) > 1e-6
    write_colored_ply(path, xyz_flat[valid], rgb_flat[valid])


def create_sphere_points(center, radius, n_u=80, n_v=40):
    center = np.asarray(center, dtype=np.float64)
    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)
    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]).astype(np.float32)


def get_pose_p_pre(pose):
    if pose is None:
        return None
    if pose.get("p_pre_mm") is not None:
        return np.asarray(pose["p_pre_mm"], dtype=np.float64)
    if pose.get("p_target_mm") is not None:
        return np.asarray(pose["p_target_mm"], dtype=np.float64)
    return None


def transform_points_for_viz(points):
    if points is None:
        return None
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    return (R @ np.asarray(points, dtype=np.float64).T).T


def transform_vector_for_viz(v):
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    v = R @ np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v if n < 1e-9 else v / n


def get_eye_points_for_visualization(xyz_crop, trocar_mask, trocar_points):
    if xyz_crop is None or trocar_mask is None or trocar_points is None or len(trocar_points) == 0:
        return None
    z_med = np.median(trocar_points[:, 2])
    points = xyz_crop[trocar_mask == 0]
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    nonzero = np.linalg.norm(points, axis=1) > 1e-6
    points = points[nonzero]
    points = points[np.abs(points[:, 2] - z_med) < 20.0]
    return points


def find_nearest_pixel_in_xyz(xyz_crop, point_3d):
    if xyz_crop is None or point_3d is None:
        return None
    valid = np.isfinite(xyz_crop).all(axis=2)
    valid &= np.linalg.norm(xyz_crop, axis=2) > 1e-6
    coords = np.argwhere(valid)
    pts = xyz_crop[valid]
    if len(pts) == 0:
        return None
    dists = np.linalg.norm(pts - np.asarray(point_3d, dtype=np.float64), axis=1)
    idx = int(np.argmin(dists))
    y, x = coords[idx]
    return int(x), int(y), float(dists[idx])


def draw_pose_on_crop(rgb_crop, xyz_crop, trocar_3d, pose, out_path):
    if rgb_crop is None or xyz_crop is None or pose is None:
        return
    vis = rgb_crop.copy()

    items = [
        (trocar_3d, "trocar_3d", (255, 255, 0), 4),
        (pose.get("p_entry_eye_mm"), "p_entry_eye", (255, 0, 0), 5),
        (pose.get("p_entry_trocar_mm"), "p_entry_trocar", (0, 255, 255), 5),
    ]
    for point, label, color, radius in items:
        px = find_nearest_pixel_in_xyz(xyz_crop, point)
        if px is None:
            continue
        x, y, _ = px
        cv2.circle(vis, (x, y), radius, color, -1)
        cv2.putText(vis, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    save_rgb_image(out_path, vis)


def plot_pose_3d(eye_points, trocar_points, pose, out_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    eye_viz = sample_points(transform_points_for_viz(eye_points), 4000)
    trocar_viz = sample_points(transform_points_for_viz(trocar_points), 1000)

    p0 = transform_points_for_viz(np.asarray(pose["p0_trocar_mm"])[None, :])[0]
    c_eye = transform_points_for_viz(np.asarray(pose["eye_center_mm"])[None, :])[0]
    p_entry_eye = transform_points_for_viz(np.asarray(pose["p_entry_eye_mm"])[None, :])[0]
    p_entry_trocar = transform_points_for_viz(np.asarray(pose["p_entry_trocar_mm"])[None, :])[0]
    p_pre = transform_points_for_viz(get_pose_p_pre(pose)[None, :])[0]
    v = transform_vector_for_viz(pose["v_trocar"])
    r_eye = float(pose["eye_radius_mm"])

    if eye_viz is not None and len(eye_viz) > 0:
        ax.scatter(eye_viz[:, 0], eye_viz[:, 1], eye_viz[:, 2], s=1, alpha=0.3, label="eye")
    if trocar_viz is not None and len(trocar_viz) > 0:
        ax.scatter(trocar_viz[:, 0], trocar_viz[:, 1], trocar_viz[:, 2], s=8, alpha=0.9, label="trocar")

    for p, label, size in [
        (c_eye, "eye center", 80),
        (p_entry_eye, "entry eye", 100),
        (p_entry_trocar, "entry trocar", 100),
        (p_pre, "p_pre", 100),
        (p0, "p0", 60),
    ]:
        ax.scatter([p[0]], [p[1]], [p[2]], s=size, label=label)

    t = np.linspace(-60, 30, 150)
    line = p0[None, :] + t[:, None] * v[None, :]
    ax.plot(line[:, 0], line[:, 1], line[:, 2], linewidth=2, label="axis")

    u = np.linspace(0, 2 * np.pi, 30)
    vv = np.linspace(0, np.pi, 20)
    xs = c_eye[0] + r_eye * np.outer(np.cos(u), np.sin(vv))
    ys = c_eye[1] + r_eye * np.outer(np.sin(u), np.sin(vv))
    zs = c_eye[2] + r_eye * np.outer(np.ones_like(u), np.cos(vv))
    ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, alpha=0.2)

    all_pts = [np.array([c_eye, p_entry_eye, p_entry_trocar, p_pre, p0])]
    if eye_viz is not None and len(eye_viz) > 0:
        all_pts.append(eye_viz)
    if trocar_viz is not None and len(trocar_viz) > 0:
        all_pts.append(trocar_viz)
    all_pts = np.vstack(all_pts)

    center = (all_pts.min(axis=0) + all_pts.max(axis=0)) / 2
    radius = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close(fig)


def create_sphere_axis_debug_ply(pose, out_path):
    c_eye = np.asarray(pose["eye_center_mm"], dtype=np.float64)
    r_eye = float(pose["eye_radius_mm"])
    p0 = np.asarray(pose["p0_trocar_mm"], dtype=np.float64)
    v_axis = np.asarray(pose["v_trocar"], dtype=np.float64)
    v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-12)

    sphere_points = create_sphere_points(c_eye, r_eye)
    sphere_colors = np.tile(np.array([[255, 220, 0]], dtype=np.uint8), (len(sphere_points), 1))

    t = np.linspace(-60, 30, 500)
    axis_points = (p0[None, :] + t[:, None] * v_axis[None, :]).astype(np.float32)
    axis_colors = np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (len(axis_points), 1))

    points = np.vstack([sphere_points, axis_points])
    colors = np.vstack([sphere_colors, axis_colors])
    write_colored_ply(out_path, points, colors)


def export_eye_sphere_debug_plys(xyz_crop, trocar_mask, trocar_points, pose, out_dir):
    eye_used = None
    if pose.get("eye_points_used") is not None:
        eye_used = np.asarray(pose["eye_points_used"], dtype=np.float32)
    else:
        eye_used = filter_points(get_eye_points_for_visualization(xyz_crop, trocar_mask, trocar_points))

    if eye_used is not None and len(eye_used) > 0:
        write_colored_ply(out_dir / "debug_eye_points_USED_FOR_SPHERE.ply", eye_used)

    c_eye = np.asarray(pose["eye_center_mm"], dtype=np.float64)
    r_eye = float(pose["eye_radius_mm"])
    p_entry_eye = np.asarray(pose["p_entry_eye_mm"], dtype=np.float64)
    p_entry_trocar = np.asarray(pose["p_entry_trocar_mm"], dtype=np.float64)
    p_pre = get_pose_p_pre(pose)
    p0 = np.asarray(pose["p0_trocar_mm"], dtype=np.float64)
    v_axis = np.asarray(pose["v_trocar"], dtype=np.float64)

    sphere_points = create_sphere_points(c_eye, r_eye)
    write_colored_ply(
        out_dir / "debug_fitted_sphere_points.ply",
        sphere_points,
        np.tile(np.array([[255, 220, 0]], dtype=np.uint8), (len(sphere_points), 1)),
    )

    t = np.linspace(-60, 30, 300)
    axis_points = p0[None, :] + t[:, None] * v_axis[None, :]

    points = []
    colors = []

    def add_cloud(cloud, rgb):
        if cloud is None or len(cloud) == 0:
            return
        points.append(np.asarray(cloud, dtype=np.float32))
        colors.append(np.tile(np.array([rgb], dtype=np.uint8), (len(cloud), 1)))

    add_cloud(eye_used, [255, 255, 255])
    add_cloud(trocar_points, [0, 255, 0])
    add_cloud(sphere_points, [255, 220, 0])
    add_cloud(axis_points, [255, 0, 255])
    add_cloud(np.tile(p_entry_eye[None, :], (100, 1)), [255, 0, 0])
    add_cloud(np.tile(p_entry_trocar[None, :], (100, 1)), [0, 255, 255])
    add_cloud(np.tile(p_pre[None, :], (100, 1)), [0, 80, 255])
    add_cloud(np.tile(c_eye[None, :], (100, 1)), [255, 165, 0])

    write_colored_ply(out_dir / "debug_COMBINED_eye_sphere_pose.ply", np.vstack(points), np.vstack(colors))


def draw_results(rgb_bgr, result):
    vis = rgb_bgr.copy()
    if result is None:
        return vis

    x1, y1, x2, y2 = result["eye_bbox"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 100, 0), 2)
    cv2.putText(vis, f"eye {result['conf_A']:.2f}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

    if result.get("trocar_global") is not None:
        tx1, ty1, tx2, ty2 = result["trocar_global"]
        cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), (0, 220, 0), 2)
        cv2.putText(vis, "trocar", (tx1, ty1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    else:
        cv2.putText(vis, "trocar not detected", (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)

    return vis


def make_result_summary(result):
    if result is None:
        return {"status": "no_eye_detection"}

    summary = {
        "status": "ok" if result.get("pose") is not None else "partial",
        "eye_bbox": result.get("eye_bbox"),
        "conf_A": result.get("conf_A"),
        "trocar_global": result.get("trocar_global"),
        "offset": result.get("offset"),
        "trocar_3d_mm": result.get("trocar_3d_mm"),
        "num_trocar_points": int(len(result["trocar_pointcloud"])) if result.get("trocar_pointcloud") is not None else 0,
        "mask_pixels": int(np.sum(result["seg_mask"] > 0)) if result.get("seg_mask") is not None else 0,
        "timings_ms": result.get("timings_ms"),
    }

    pose = result.get("pose")
    if pose is not None:
        p_pre = get_pose_p_pre(pose)
        summary["pose"] = {
            "p0_trocar_mm": np.asarray(pose["p0_trocar_mm"]).tolist(),
            "v_trocar": np.asarray(pose["v_trocar"]).tolist(),
            "z_axis_cam": np.asarray(pose.get("z_axis_cam", pose["v_trocar"])).tolist(),
            "eye_center_mm": np.asarray(pose["eye_center_mm"]).tolist(),
            "eye_radius_mm": float(pose["eye_radius_mm"]),
            "p_entry_eye_mm": np.asarray(pose["p_entry_eye_mm"]).tolist(),
            "p_entry_trocar_mm": np.asarray(pose["p_entry_trocar_mm"]).tolist(),
            "p_pre_mm": p_pre.tolist(),
            "trocar_entry_offset_mm": float(pose.get("trocar_entry_offset_mm", 1.0)),
            "target_clearance_mm": float(pose.get("target_clearance_mm", 50.0)),
            "num_trocar_points": int(pose.get("num_trocar_points", 0)),
            "num_eye_points": int(pose.get("num_eye_points", 0)),
            "num_sphere_inliers": int(pose.get("num_sphere_inliers", 0)),
            "robot_motion_allowed": bool(pose.get("robot_motion_allowed", False)),
            "robot_motion_block_reason": pose.get("robot_motion_block_reason"),
        }
        if pose.get("T_cam_target") is not None:
            summary["pose"]["T_cam_target"] = np.asarray(pose["T_cam_target"]).tolist()
        if pose.get("T_base_target") is not None:
            summary["pose"]["T_base_target"] = np.asarray(pose["T_base_target"]).tolist()

    return summary


def make_prediction_json(result):
    if result is None or result.get("pose") is None:
        prediction = {"status": "no_pose", "pose": None}
        if result is not None and result.get("timings_ms") is not None:
            prediction["timings_ms"] = result.get("timings_ms")
        return prediction

    pose = result["pose"]
    p_pre = get_pose_p_pre(pose)

    prediction = {
        "status": "ok",
        "frame": "camera_mm",
        "units": "mm",
        "timings_ms": result.get("timings_ms"),
        "pose": {
            "p_entry_trocar_mm": np.asarray(pose["p_entry_trocar_mm"]).tolist(),
            "p_entry_eye_mm": np.asarray(pose["p_entry_eye_mm"]).tolist(),
            "p_pre_mm": p_pre.tolist(),
            "v_trocar": np.asarray(pose["v_trocar"]).tolist(),
            "z_axis_cam": np.asarray(pose.get("z_axis_cam", pose["v_trocar"])).tolist(),
            "eye_center_mm": np.asarray(pose["eye_center_mm"]).tolist(),
            "eye_radius_mm": float(pose["eye_radius_mm"]),
            "p_visible_eye_mm": np.asarray(pose["p_visible_eye_mm"]).tolist() if pose.get("p_visible_eye_mm") is not None else None,
            "n_anterior": np.asarray(pose["n_anterior"]).tolist() if pose.get("n_anterior") is not None else None,
            "p_posterior_pole_mm": np.asarray(pose["p_posterior_pole_mm"]).tolist() if pose.get("p_posterior_pole_mm") is not None else None,
            "p0_trocar_mm": np.asarray(pose["p0_trocar_mm"]).tolist() if pose.get("p0_trocar_mm") is not None else None,
            "num_trocar_points": int(pose.get("num_trocar_points", 0)),
            "num_eye_points": int(pose.get("num_eye_points", 0)),
            "num_sphere_inliers": int(pose.get("num_sphere_inliers", 0)),
            "trocar_entry_offset_mm": float(pose.get("trocar_entry_offset_mm", 1.0)),
            "target_clearance_mm": float(pose.get("target_clearance_mm", 50.0)),
        },
    }

    if pose.get("T_cam_target") is not None:
        prediction["pose"]["T_cam_target"] = np.asarray(pose["T_cam_target"]).tolist()
    if pose.get("T_base_target") is not None:
        prediction["pose"]["T_base_target"] = np.asarray(pose["T_base_target"]).tolist()

    prediction.update(prediction["pose"])
    return prediction


def save_evaluation_prediction(result, out_dir):
    with open(out_dir / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(make_prediction_json(result), f, indent=2)


def save_debug_outputs(capture_name, rgb, xyz, result, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    save_rgb_image(out_dir / f"{capture_name}_rgb.png", rgb)

    if xyz is not None:
        np.save(out_dir / f"{capture_name}_xyz.npy", xyz.astype(np.float32))
        write_rgb_xyz_ply(out_dir / f"{capture_name}_captured_cloud.ply", xyz, rgb)

    with open(out_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_result_summary(result), f, indent=2)
    save_evaluation_prediction(result, out_dir)

    if result is None:
        return

    eye_bbox = result.get("eye_bbox")
    if eye_bbox is not None:
        vis_eye = rgb.copy()
        x1, y1, x2, y2 = [int(v) for v in eye_bbox]
        cv2.rectangle(vis_eye, (x1, y1), (x2, y2), (255, 100, 0), 2)
        save_rgb_image(out_dir / "debug_01_eye_bbox.png", vis_eye)

    rgb_crop = result.get("rgb_crop")
    xyz_crop = result.get("xyz_crop")
    seg_mask = result.get("seg_mask")
    trocar_points = result.get("trocar_pointcloud")
    trocar_3d = result.get("trocar_3d_mm")
    pose = result.get("pose")

    if rgb_crop is not None:
        save_rgb_image(out_dir / "debug_02_eye_crop.png", rgb_crop)
    if xyz_crop is not None:
        np.save(out_dir / "debug_xyz_crop.npy", xyz_crop.astype(np.float32))
    if seg_mask is not None:
        cv2.imwrite(str(out_dir / "debug_03_trocar_mask_crop.png"), (seg_mask > 0).astype(np.uint8) * 255)
        save_mask_overlay(rgb_crop, seg_mask, out_dir / "debug_04_eye_crop_plus_trocar_mask.png")

    if pose is None or trocar_points is None or xyz_crop is None or seg_mask is None:
        return

    eye_points = np.asarray(pose["eye_points_used"], dtype=np.float32) if pose.get("eye_points_used") is not None else get_eye_points_for_visualization(xyz_crop, seg_mask, trocar_points)

    draw_pose_on_crop(rgb_crop, xyz_crop, trocar_3d, pose, out_dir / "debug_05_pose_on_crop.png")
    plot_pose_3d(eye_points, trocar_points, pose, out_dir / "debug_06_pose_3d.png")
    create_sphere_axis_debug_ply(pose, out_dir / "debug.ply")
    export_eye_sphere_debug_plys(xyz_crop, seg_mask, trocar_points, pose, out_dir)

    if pose.get("T_cam_target") is not None:
        np.save(out_dir / "T_cam_target.npy", np.asarray(pose["T_cam_target"], dtype=np.float64))
    if pose.get("T_base_target") is not None:
        np.save(out_dir / "T_base_target.npy", np.asarray(pose["T_base_target"], dtype=np.float64))

    with open(out_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_result_summary(result), f, indent=2)
    save_evaluation_prediction(result, out_dir)


# ============================================================
# Robot helpers
# ============================================================

if ROS2_AVAILABLE:
    class JointTargetNode(Node):
        def __init__(self):
            super().__init__("safe_joint_prepose_node")
            self.latest_joints_deg = None
            self.pub = self.create_publisher(Float64MultiArray, JOINT_TARGET_TOPIC, 10)
            self.sub = self.create_subscription(JointState, "/joint_states", self._cb_joint_states, 10)

        def _cb_joint_states(self, msg: JointState):
            if len(msg.position) < 6:
                return
            q = np.degrees(np.asarray(msg.position[:6], dtype=np.float64))
            if msg.name and all(name in msg.name for name in MECA_JOINT_NAMES):
                idx = [msg.name.index(name) for name in MECA_JOINT_NAMES]
                q = np.degrees(np.asarray([msg.position[i] for i in idx], dtype=np.float64))
            self.latest_joints_deg = q.tolist()

        def wait_current_joints(self, timeout_sec: float):
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.latest_joints_deg is not None:
                    return self.latest_joints_deg
            return None

        def publish_joints(self, joints_deg):
            deadline = time.time() + max(2.0, SAFE_JOINT_DISCOVERY_WAIT_SEC)
            while self.pub.get_subscription_count() == 0 and time.time() < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)
            if self.pub.get_subscription_count() == 0:
                raise RuntimeError(f"No subscriber found on {JOINT_TARGET_TOPIC}")

            msg = Float64MultiArray()
            msg.data = [float(v) for v in joints_deg]
            for _ in range(3):
                self.pub.publish(msg)
                rclpy.spin_once(self, timeout_sec=0.1)


def get_script_path(path: Path):
    if path.exists():
        return path
    candidate = Path(__file__).resolve().parent / path.name
    return candidate if candidate.exists() else path


def confirm_robot_step(title: str, details=None) -> bool:
    print(f"\n[CONFIRM] {title}")

    if VERBOSE_TERMINAL and details is not None:
        lines = details if isinstance(details, (list, tuple)) else [details]
        for line in lines:
            print(f"  {line}")

    ans = input("ENTER = execute | s = skip | q = abort: ").strip().lower()

    if ans in ("q", "quit", "exit"):
        raise KeyboardInterrupt(f"User aborted before: {title}")
    if ans in ("s", "skip"):
        print(f"  Skipped: {title}")
        return False

    print(f"  Executing: {title}")
    return True

def resolve_eye_side_once():
    if SAFE_POSE_EYE_SIDE in ("right", "left"):
        return SAFE_POSE_EYE_SIDE

    while True:
        ans = input("Eye side for safe pre-pose [r=right, l=left]: ").strip().lower()
        if ans in ("r", "right"):
            return "right"
        if ans in ("l", "left"):
            return "left"
        print("Use 'r' or 'l'.")


def safe_joints_for_side(eye_side: str):
    if eye_side == "right":
        return SAFE_JOINTS_RIGHT_EYE_DEG
    if eye_side == "left":
        return SAFE_JOINTS_LEFT_EYE_DEG
    raise ValueError(f"Invalid eye side: {eye_side}")


def send_safe_joint_prepose(eye_side: str):
    if not ROS2_AVAILABLE:
        raise RuntimeError(f"ROS 2 imports are not available: {ROS2_IMPORT_ERROR}")

    final_joints = safe_joints_for_side(eye_side)

    if REQUIRE_ENTER_BEFORE_SAFE_JOINTS:
        allowed = confirm_robot_step(
            "Safe joint pre-pose",
            details=[
                f"eye_side={eye_side}",
                f"final_joints_deg={final_joints}",
                "no MoveIt is used for this step",
            ],
        )
        if not allowed:
            return False

    if not rclpy.ok():
        rclpy.init(args=None)

    node = JointTargetNode()
    try:
        if SAFE_PREPOSE_TWO_STEP_BEFORE_J1:
            current = node.wait_current_joints(SAFE_CURRENT_J1_TIMEOUT_SEC)
            if current is None:
                raise RuntimeError("Could not read current /joint_states")
            step1 = [float(current[0]), 0.0, 0.0, 0.0, 49.0, 0.0]
            log_verbose(f"  Safe pre-pose step 1: {step1}")
            node.publish_joints(step1)
            time.sleep(SAFE_JOINT_SETTLE_SEC)

        log_verbose(f"  Safe pre-pose step 2: {final_joints}")
        node.publish_joints(final_joints)
        time.sleep(SAFE_JOINT_SETTLE_SEC)
        return True
    finally:
        node.destroy_node()


def build_robot_target_from_prediction(capture_dir: Path):
    prediction_path = capture_dir / "prediction.json"
    target_path = capture_dir / "T_base_target.npy"
    script = get_script_path(MECA_POSE_SCRIPT)

    cmd = ["python3", str(script), "--prediction", str(prediction_path), "--out-dir", str(capture_dir)]
    log_verbose("  Building robot target from prediction.")
    run_command(cmd)

    if not target_path.exists():
        raise FileNotFoundError(f"Target was not created: {target_path}")

    return target_path, np.load(target_path).astype(np.float64)

def send_native_movepose(target_path: Path):
    script = get_script_path(NATIVE_MOVEPOSE_SCRIPT)
    cmd = [
        "python3",
        str(script),
        "--target",
        str(target_path),
        "--extra-clearance-mm",
        str(NATIVE_MOVEPOSE_EXTRA_CLEARANCE_MM),
    ]
    if NATIVE_MOVEPOSE_DRY_RUN:
        cmd.append("--dry-run")

    if REQUIRE_ENTER_BEFORE_NATIVE_MOVEPOSE:
        allowed = confirm_robot_step(
            "Native Meca500 MovePose to T_base_target",
            details=[f"target={target_path}"],
        )
        if not allowed:
            return False

    run_command(cmd)
    return not NATIVE_MOVEPOSE_DRY_RUN

def send_movelin_advance(distance_mm: float, title: str, require_enter: bool):
    script = get_script_path(MOVELIN_ADVANCE_SCRIPT)
    cmd = [
        "python3",
        str(script),
        "--distance-mm",
        str(distance_mm),
        "--axis-column",
        str(ADVANCE_AXIS_COLUMN),
        "--axis-sign",
        str(ADVANCE_AXIS_SIGN),
    ]

    if require_enter:
        allowed = confirm_robot_step(
            title,
            details=[
                f"distance_mm={distance_mm:.3f}",
                f"axis_column={ADVANCE_AXIS_COLUMN}",
                f"axis_sign={ADVANCE_AXIS_SIGN:+.0f}",
            ],
        )
        if not allowed:
            return False

    run_command(cmd)
    return True

def run_j6_alignment():
    if not AUTO_J6_ROTATION_STEP:
        return True
    script = get_script_path(J6_ROTATE_SCRIPT)
    print("\n[MANUAL] Interactive J6 alignment")
    proc = run_command(["python3", str(script)], interactive=True, check=False)
    return proc.returncode == 0

def execute_robot_sequence(target_path: Path, eye_side: str):
    safe_ok = send_safe_joint_prepose(eye_side)
    if not safe_ok:
        return False

    if not AUTO_NATIVE_MOVEPOSE:
        return True

    move_pose_ok = send_native_movepose(target_path)
    if not move_pose_ok:
        return False

    advance_ok = True
    if AUTO_ADVANCE_MOVELIN:
        advance_ok = send_movelin_advance(
            ADVANCE_MM,
            "MoveLin advance from current TCP",
            REQUIRE_ENTER_BEFORE_MOVELIN_ADVANCE,
        )

    if not advance_ok:
        return False

    if not run_j6_alignment():
        print("  J6 alignment cancelled. Final MoveLin skipped.")
        return False

    if AUTO_FINAL_MOVELIN:
        return send_movelin_advance(
            FINAL_MOVELIN_MM,
            f"Final MoveLin insertion {FINAL_MOVELIN_MM:.1f} mm",
            REQUIRE_ENTER_BEFORE_FINAL_MOVELIN,
        )

    return True


# ============================================================
# Camera and main loop
# ============================================================

def capture_frame(camera, settings):
    with camera.capture_2d_3d(settings) as frame:
        pc = frame.point_cloud()
        try:
            rgba = pc.copy_data("rgba_srgb")
        except Exception:
            rgba = pc.copy_data("rgba")
        rgb = rgba[:, :, :3].astype(np.uint8)
        xyz = pc.copy_data("xyz").astype(np.float32)
    return rgb, xyz


def print_result(result):
    if result is None:
        print("  Vision: eye=NO | trocar=NO | pose=NO")
        return

    eye_ok = result.get("eye_bbox") is not None
    trocar_ok = result.get("trocar_global") is not None
    pose_ok = result.get("pose") is not None

    print(
        "  Vision: "
        f"eye={'OK' if eye_ok else 'NO'} | "
        f"trocar={'OK' if trocar_ok else 'NO'} | "
        f"pose={'OK' if pose_ok else 'NO'}"
    )

    if not VERBOSE_TERMINAL:
        return

    if eye_ok:
        print(f"  eye_bbox: {result['eye_bbox']}  conf={result['conf_A']}")
    if trocar_ok:
        print(f"  trocar_bbox: {result['trocar_global']}")
        if result.get("trocar_3d_mm") is not None:
            x, y, z = result["trocar_3d_mm"]
            print(f"  trocar_3d: X={x:.2f} Y={y:.2f} Z={z:.2f} mm")

    pose = result.get("pose")
    if pose is None:
        return

    p_eye = pose["p_entry_eye_mm"]
    p_trocar = pose["p_entry_trocar_mm"]
    p_pre = get_pose_p_pre(pose)
    v = pose["v_trocar"]
    c = pose["eye_center_mm"]
    r = pose["eye_radius_mm"]

    print(f"  p_entry_eye:    X={p_eye[0]:.2f} Y={p_eye[1]:.2f} Z={p_eye[2]:.2f} mm")
    print(f"  p_entry_trocar: X={p_trocar[0]:.2f} Y={p_trocar[1]:.2f} Z={p_trocar[2]:.2f} mm")
    print(f"  p_pre:          X={p_pre[0]:.2f} Y={p_pre[1]:.2f} Z={p_pre[2]:.2f} mm")
    print(f"  axis:           vx={v[0]:.3f} vy={v[1]:.3f} vz={v[2]:.3f}")
    print(f"  eye:            C=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) r={r:.2f} mm")

def process_valid_result(result, capture_dir: Path, eye_side: str):
    pose = result["pose"]

    if ENABLE_POSE_SAFETY_CHECKS:
        motion_allowed, block_reason = validate_pose_for_robot(pose)
    else:
        motion_allowed, block_reason = True, "pose safety checks disabled"

    pose["robot_motion_allowed"] = bool(motion_allowed)
    pose["robot_motion_block_reason"] = block_reason

    print(f"  Safety: {'OK' if motion_allowed else 'BLOCKED'} ({block_reason})")

    save_evaluation_prediction(result, capture_dir)
    target_path, T_base_target = build_robot_target_from_prediction(capture_dir)
    pose["T_base_target"] = T_base_target

    print(
        "  Target: "
        f"X={T_base_target[0, 3]:.2f} "
        f"Y={T_base_target[1, 3]:.2f} "
        f"Z={T_base_target[2, 3]:.2f} mm"
    )

    if not motion_allowed:
        print("  Robot: skipped by safety check")
        return None

    t_robot = time.perf_counter()
    try:
        executed = execute_robot_sequence(target_path, eye_side)
        print(f"  Robot: {'completed' if executed else 'not completed'}")
    except subprocess.CalledProcessError as e:
        print(f"  Robot: command failed with exit code {e.returncode}")
    return elapsed_ms(t_robot)

def main():
    eye_side = resolve_eye_side_once()

    model_A = YOLO(str(MODEL_A_PATH))
    model_B = load_model_b(MODEL_B_PATH)
    print(f"Device: {DEVICE}")

    app = zivid.Application()
    print("Connecting to camera...")
    camera = app.connect_camera()
    print(f"Camera ready: {camera.info.model_name} | Serial: {camera.info.serial_number}\n")

    settings = zivid.Settings.load(str(SETTINGS_YML))

    base_name = input("Capture name (or ENTER for 'test'): ").strip() or "test"
    input("\nPress ENTER to start...\n")

    for i in range(1, 999):
        capture_name = f"{base_name}_{i:03d}"
        capture_dir = OUT_DIR / capture_name
        capture_dir.mkdir(parents=True, exist_ok=True)

        retries = 0
        valid_done = False
        last_rgb = None
        last_xyz = None
        last_result = None

        while not valid_done:
            retry_msg = f" [retry {retries}/{MAX_RETRIES}]" if retries else ""
            print(f"\nCapture: {capture_name}{retry_msg}")

            rgb, xyz = capture_frame(camera, settings)
            last_rgb, last_xyz = rgb, xyz

            t_total = time.perf_counter()
            result = run_pipeline(rgb, xyz, model_A, model_B)
            robot_time_ms = None

            print_result(result)

            if result is not None and result.get("pose") is not None:
                try:
                    robot_time_ms = process_valid_result(result, capture_dir, eye_side)
                except Exception as e:
                    print(f"  Robot target/execution skipped: {e}")

            total_ms = elapsed_ms(t_total)
            if result is not None:
                timings = result.setdefault("timings_ms", {})
                timings["robot_motion_time_ms"] = robot_time_ms
                timings["total_pipeline_time_ms"] = total_ms

            last_result = result
            print_pipeline_timings(result.get("timings_ms") if result is not None else {"total_pipeline_time_ms": total_ms})
            print()

            valid_done = result is not None and result.get("trocar_global") is not None and result.get("pose") is not None

            if not valid_done:
                if retries < MAX_RETRIES:
                    retries += 1
                    print(f"  Invalid result. Retrying ({retries}/{MAX_RETRIES})...\n")
                    continue
                print(f"  Valid result not obtained after {MAX_RETRIES} retries.\n")
                break

        if last_rgb is not None:
            vis = draw_results(cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR), last_result)
            cv2.imwrite(str(capture_dir / f"{capture_name}_result.png"), vis)
            save_debug_outputs(capture_name, last_rgb, last_xyz, last_result, capture_dir)
            print(f"  Saved: {capture_dir}")

        ans = input("ENTER = next | q = quit: ").strip().lower()
        if ans == "q":
            break

    if ROS2_AVAILABLE and rclpy.ok():
        rclpy.shutdown()

    print("\nDone.")


if __name__ == "__main__":
    main()
