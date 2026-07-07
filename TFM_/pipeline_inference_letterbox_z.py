from pathlib import Path
import time
import json

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import zivid
from ultralytics import YOLO

from crop_utils import (
    crop_rgb_and_xyz,
    bbox_local_to_global,
    get_3d_from_mask,
)

#from pose_utils import estimate_trocar_pose, filter_points
from pose_utils_trocarfit import estimate_trocar_pose, filter_points

import subprocess

# ─────────────────────────────────────────────
# OPTIONAL ROS 2 / RVIZ MARKERS
# ─────────────────────────────────────────────
# The inference pipeline can still run without ROS 2 Python imports.
# If rclpy is available, the script publishes target-pose markers before
# calling the MoveIt plan/execute script.
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    ROS2_MARKERS_AVAILABLE = True
except Exception as _ros_import_error:
    rclpy = None
    Node = object
    Marker = None
    Point = None
    QoSProfile = None
    ReliabilityPolicy = None
    DurabilityPolicy = None
    ROS2_MARKERS_AVAILABLE = False
    ROS2_MARKERS_IMPORT_ERROR = _ros_import_error

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SETTINGS_YML = Path("ZIVID/Z2p_M60_Inspection_SmallFeatures_Off.yml")

MODEL_A_PATH = Path("models/model_A_yolo26s.pt")          # YOLO26s
MODEL_B_PATH = Path("models/model_b_letterbox_best.pt")   # U-Net + ResNet34 + letterbox

MARGIN = 0.12
CONF_A = 0.5
SEG_THRESH = 0.5

# Must match the value used during Model B training.
IMG_SIZE = 320

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_RETRIES = 3

OUT_DIR = Path(r"Inference")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MOVEIT_PLAN_SCRIPT = Path("/home/lucia/ros2_ws/scripts/plan_to_trocar_ros2.py")

# Second-stage script: converts p_pre_cam + z_axis_cam to T_base_target
# using calibration and the supervisor q_align * default_orientation method.
MECA_POSE_SCRIPT = Path("build_meca_pose_from_axis_z.py")

# If True, the pipeline calls plan_to_trocar_ros2.py after T_base_target.npy is created.
# plan_to_trocar_ros2.py sends the target to MoveIt. If MOVEIT_PLAN_ONLY is False,
# MoveIt will execute and internally call /meca_arm_controller/follow_joint_trajectory.
AUTO_PLAN = True

# Safety switch.
#   True  -> only plan in MoveIt; do not move the robot.
#   False -> plan + execute through MoveIt.
# For the real robot, test first with True, then change to False.
MOVEIT_PLAN_ONLY = False

# Conservative MoveIt execution scaling.
MOVEIT_VELOCITY_SCALING = 0.05
MOVEIT_ACCELERATION_SCALING = 0.05

# Hybrid execution strategy:
#   - T_base_target.npy is the native final target at 50 mm clearance.
#   - MoveIt plans to T_stage = T_base_target moved backwards by this value
#     along -Z of T_base_target. With 30 mm here and 50 mm in pose_utils,
#     MoveIt stops at an 80 mm clearance stage.
MOVEIT_PRE_CLEARANCE_MM = 30.0

# Native Meca500 MovePose to the final T_base_target after MoveIt reaches T_stage.
# This bypasses the URDF FK discrepancy for the final Cartesian approach.
AUTO_NATIVE_MOVEPOSE = True
NATIVE_MOVEPOSE_SCRIPT = Path("/home/lucia/ros2_ws/scripts/send_meca_move_pose_from_target.py")
NATIVE_MOVEPOSE_EXTRA_CLEARANCE_MM = 0.0
NATIVE_MOVEPOSE_DRY_RUN = False

# Orientation correction switch passed to plan_to_trocar_ros2.py.
#   False -> apply the Meca500/MoveIt axis correction in the planning script.
#   True  -> pass --disable-axis-correction for debugging double-correction issues.
# Test both in RViz with MOVEIT_PLAN_ONLY=True if the TCP orientation looks wrong.
MOVEIT_DISABLE_AXIS_CORRECTION = False

# Optional final insertion/advance after native MovePose reaches T_base_target.
# This publishes a relative native Meca500 MoveLin command from the CURRENT TCP pose.
# Keep False until MovePose to T_target is validated in the air.
AUTO_ADVANCE_MOVELIN = True
ADVANCE_MM = 45.0
ADVANCE_AXIS_COLUMN = 2      # 2 = current TCP +Z axis from GetPose/SetTrf(0,0,L)
ADVANCE_AXIS_SIGN = 1.0      # +1 should move towards p_entry if local +Z is insertion axis
MOVELIN_ADVANCE_SCRIPT = Path("/home/lucia/ros2_ws/scripts/move_lin_advance_current.py")

# Manual safety confirmations before robot-moving stages.
# With these enabled, the pipeline pauses and asks for ENTER before:
#   1) MoveIt -> T_stage
#   2) optional MoveLin advance (primera inserción)
#   3) MoveLin final (después de la alineación J6)
# MovePose se ejecuta automáticamente después de que MoveIt completa (sin ENTER).
# At each prompt:
#   ENTER = execute this step
#   s     = skip this step
#   q     = abort the pipeline immediately
REQUIRE_ENTER_BEFORE_MOVEIT = True
REQUIRE_ENTER_BEFORE_NATIVE_MOVEPOSE = False   # auto-ejecuta tras MoveIt
REQUIRE_ENTER_BEFORE_MOVELIN_ADVANCE = True

# Paso interactivo de rotación J6 para alinear la aguja, ejecutado tras el primer MoveLin.
# El operador puede rotar J6 libremente desde la terminal hasta encontrar la mejor orientación.
# Poner a False para deshabilitar este paso.
AUTO_J6_ROTATION_STEP = True
J6_ROTATE_SCRIPT = Path("/home/lucia/ros2_ws/scripts/rotate_j6_interactive.py")

# MoveLin final de FINAL_MOVELIN_MM mm, ejecutado tras la alineación J6.
AUTO_FINAL_MOVELIN_5MM = True
FINAL_MOVELIN_MM = 5.0
REQUIRE_ENTER_BEFORE_FINAL_MOVELIN = True

# RViz marker visualization.
# These markers are only a visual preview of the target pose and tool axes.
# The actual trajectory is computed/executed by MoveIt via MOVEIT_PLAN_SCRIPT.
ENABLE_RVIZ_MARKERS = True
RVIZ_MARKER_TOPIC = "/meca_target_pose_markers"
RVIZ_FRAME_ID = "meca_base_link"
RVIZ_AXIS_LENGTH_MM = 60.0
RVIZ_AXIS_BACK_MM = 80.0

# ─────────────────────────────────────────────
# TIMING HELPERS
# ─────────────────────────────────────────────

def sync_cuda_if_needed():
    """
    Synchronize CUDA operations so GPU timings are more realistic.
    Without this, PyTorch/YOLO calls can look artificially fast because
    CUDA kernels are asynchronous.
    """
    if torch.cuda.is_available() and DEVICE == "cuda":
        torch.cuda.synchronize()


def elapsed_ms(t_start):
    """Return elapsed time in milliseconds from a perf_counter timestamp."""
    return (time.perf_counter() - t_start) * 1000.0


def format_ms(value):
    """Format a timing value for terminal output."""
    if value is None:
        return "N/A"
    return f"{float(value):.1f} ms"


def print_pipeline_timings(timings):
    """
    Print the main timing breakdown for one capture attempt.
    """
    timings = timings or {}

    rows = [
        ("Eye detection", "eye_detection_ms"),
        ("Eye crop generation", "eye_crop_generation_ms"),
        ("Trocar segmentation", "trocar_segmentation_ms"),
        ("Pose estimation", "pose_estimation_ms"),
        ("Planning time", "planning_time_ms"),
        ("Total pipeline time", "total_pipeline_time_ms"),
    ]

    print("  timing breakdown:")
    for label, key in rows:
        print(f"  {label:<24} {format_ms(timings.get(key))}")



# ─────────────────────────────────────────────
# RVIZ TARGET POSE MARKERS
# ─────────────────────────────────────────────

def make_rviz_visualizer():
    """
    Create the RViz marker publisher if ROS 2 is available.

    Returns:
        TargetPoseVisualizer or None.
    """
    if not ENABLE_RVIZ_MARKERS:
        return None

    if not ROS2_MARKERS_AVAILABLE:
        print(
            "  [RViz] Markers disabled: ROS 2 Python imports are not available "
            f"({ROS2_MARKERS_IMPORT_ERROR})"
        )
        return None

    if not rclpy.ok():
        rclpy.init(args=None)

    return TargetPoseVisualizer()


if ROS2_MARKERS_AVAILABLE:
    class TargetPoseVisualizer(Node):
        """
        Publish a compact RViz preview of T_base_target.

        The markers show:
          - target point
          - tool-frame X/Y/Z axes
          - tool Z axis reference line
          - text label with target coordinates

        Important:
          These markers do NOT represent the real MoveIt trajectory.
          They only show the target pose that will be passed to MoveIt.
        """

        def __init__(self):
            super().__init__("pipeline_target_pose_visualizer")

            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.RELIABLE
            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

            self.marker_pub = self.create_publisher(
                Marker,
                RVIZ_MARKER_TOPIC,
                qos,
            )

            self.get_logger().info(
                f"Publishing RViz markers on {RVIZ_MARKER_TOPIC} "
                f"in frame {RVIZ_FRAME_ID}"
            )

        @staticmethod
        def _point_mm_to_msg(p_mm):
            p = Point()
            p.x = float(p_mm[0]) / 1000.0
            p.y = float(p_mm[1]) / 1000.0
            p.z = float(p_mm[2]) / 1000.0
            return p

        @staticmethod
        def _normalize(v):
            v = np.asarray(v, dtype=np.float64)
            n = np.linalg.norm(v)
            if n < 1e-9:
                return v
            return v / n

        def _base_marker(self, marker_id, marker_type, ns="target_pose"):
            marker = Marker()
            marker.header.frame_id = RVIZ_FRAME_ID
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = int(marker_id)
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            return marker

        def _make_arrow(self, marker_id, start_mm, end_mm, color_rgba, shaft_diam=0.006, head_diam=0.018):
            marker = self._base_marker(marker_id, Marker.ARROW)
            marker.points = [
                self._point_mm_to_msg(start_mm),
                self._point_mm_to_msg(end_mm),
            ]
            marker.scale.x = shaft_diam
            marker.scale.y = head_diam
            marker.scale.z = head_diam
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color_rgba
            return marker

        def publish_target_pose(self, T_base_target: np.ndarray):
            T_base_target = np.asarray(T_base_target, dtype=np.float64)

            if T_base_target.shape != (4, 4):
                raise ValueError(f"T_base_target must be 4x4, got {T_base_target.shape}")

            p_target_mm = T_base_target[:3, 3]
            R_target = T_base_target[:3, :3]

            x_axis = self._normalize(R_target[:, 0])
            y_axis = self._normalize(R_target[:, 1])
            z_axis = self._normalize(R_target[:, 2])

            axis_len = float(RVIZ_AXIS_LENGTH_MM)
            axis_back = float(RVIZ_AXIS_BACK_MM)

            markers = []

            # Clear previous preview markers.
            clear_marker = Marker()
            clear_marker.header.frame_id = RVIZ_FRAME_ID
            clear_marker.header.stamp = self.get_clock().now().to_msg()
            clear_marker.ns = "target_pose"
            clear_marker.id = 999
            clear_marker.action = Marker.DELETEALL
            markers.append(clear_marker)

            # Target point: red sphere.
            sphere = self._base_marker(0, Marker.SPHERE)
            sphere.pose.position = self._point_mm_to_msg(p_target_mm)
            sphere.scale.x = 0.012
            sphere.scale.y = 0.012
            sphere.scale.z = 0.012
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            sphere.color.a = 1.0
            markers.append(sphere)

            # Tool axes: X red, Y green, Z blue.
            markers.append(self._make_arrow(
                1,
                p_target_mm,
                p_target_mm + axis_len * x_axis,
                (1.0, 0.0, 0.0, 1.0),
            ))
            markers.append(self._make_arrow(
                2,
                p_target_mm,
                p_target_mm + axis_len * y_axis,
                (0.0, 1.0, 0.0, 1.0),
            ))
            markers.append(self._make_arrow(
                3,
                p_target_mm,
                p_target_mm + axis_len * z_axis,
                (0.0, 0.2, 1.0, 1.0),
                shaft_diam=0.008,
                head_diam=0.022,
            ))

            # Orange line along the tool Z axis.
            # This is a pose/axis reference, not the actual MoveIt trajectory.
            axis_line = self._base_marker(4, Marker.LINE_STRIP)
            axis_line.points = [
                self._point_mm_to_msg(p_target_mm - axis_back * z_axis),
                self._point_mm_to_msg(p_target_mm + axis_len * z_axis),
            ]
            axis_line.scale.x = 0.004
            axis_line.color.r = 1.0
            axis_line.color.g = 0.5
            axis_line.color.b = 0.0
            axis_line.color.a = 1.0
            markers.append(axis_line)

            # Text label.
            text = self._base_marker(5, Marker.TEXT_VIEW_FACING)
            label_pos = p_target_mm + np.array([0.0, 0.0, 35.0], dtype=np.float64)
            text.pose.position = self._point_mm_to_msg(label_pos)
            text.scale.z = 0.018
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = (
                "T_base_target\n"
                f"X={p_target_mm[0]:.1f} mm\n"
                f"Y={p_target_mm[1]:.1f} mm\n"
                f"Z={p_target_mm[2]:.1f} mm"
            )
            markers.append(text)

            for _ in range(5):
                for marker in markers:
                    self.marker_pub.publish(marker)
                rclpy.spin_once(self, timeout_sec=0.05)

            self.get_logger().info(
                "RViz target markers published: "
                f"X={p_target_mm[0]:.1f}, "
                f"Y={p_target_mm[1]:.1f}, "
                f"Z={p_target_mm[2]:.1f} mm"
            )


        def _delete_marker(self, marker_id, ns="target_pose"):
            """Delete one marker by namespace/id without clearing other namespaces."""
            marker = Marker()
            marker.header.frame_id = RVIZ_FRAME_ID
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = int(marker_id)
            marker.action = Marker.DELETE
            return marker

        def publish_eye_and_axis(self, pose: dict, T_base_cam: np.ndarray):
            """
            Publish eye sphere + trocar/insertion axis in meca_base_link.

            pose fields are in camera frame, in millimetres.
            T_base_cam maps camera-frame millimetres to meca_base_link millimetres.
            This does not depend on pose['T_cam_target'].
            """
            T_base_cam = np.asarray(T_base_cam, dtype=np.float64)

            if T_base_cam.shape != (4, 4):
                raise ValueError(f"T_base_cam must be 4x4, got {T_base_cam.shape}")

            required = [
                "eye_center_mm",
                "eye_radius_mm",
                "p0_trocar_mm",
                "v_trocar",
            ]
            missing = [k for k in required if pose.get(k) is None]
            if missing:
                raise KeyError(f"pose is missing required keys for eye_debug: {missing}")

            def to_base_m(p_cam_mm):
                p = np.array([*p_cam_mm, 1.0], dtype=np.float64)
                return (T_base_cam @ p)[:3] / 1000.0

            def rot_to_base(v_cam):
                return T_base_cam[:3, :3] @ np.asarray(v_cam, dtype=np.float64)

            c_eye_m = to_base_m(pose["eye_center_mm"])
            r_eye_m = float(pose["eye_radius_mm"]) / 1000.0
            p0_m = to_base_m(pose["p0_trocar_mm"])

            v_axis = rot_to_base(pose["v_trocar"])
            v_axis = v_axis / (np.linalg.norm(v_axis) + 1e-12)

            markers = []

            # Delete only eye_debug markers from the previous capture.
            # Do not use DELETEALL here, because that can remove target_pose markers too.
            for marker_id in (20, 21, 22, 23, 24):
                markers.append(self._delete_marker(marker_id, ns="eye_debug"))

            # Eye sphere.
            sphere = self._base_marker(20, Marker.SPHERE, ns="eye_debug")
            sphere.pose.position.x = float(c_eye_m[0])
            sphere.pose.position.y = float(c_eye_m[1])
            sphere.pose.position.z = float(c_eye_m[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 2.0 * r_eye_m
            sphere.scale.y = 2.0 * r_eye_m
            sphere.scale.z = 2.0 * r_eye_m
            sphere.color.r = 1.0
            sphere.color.g = 0.85
            sphere.color.b = 0.0
            sphere.color.a = 0.45
            markers.append(sphere)

            # Trocar / insertion axis as magenta line.
            axis_line = self._base_marker(21, Marker.LINE_STRIP, ns="eye_debug")
            axis_line.points = []
            for t in np.linspace(-0.08, 0.04, 20):
                pt = Point()
                q = p0_m + t * v_axis
                pt.x = float(q[0])
                pt.y = float(q[1])
                pt.z = float(q[2])
                axis_line.points.append(pt)
            axis_line.scale.x = 0.004
            axis_line.color.r = 1.0
            axis_line.color.g = 0.0
            axis_line.color.b = 1.0
            axis_line.color.a = 1.0
            markers.append(axis_line)

            # Entry point if available: small red sphere.
            if pose.get("p_entry_eye_mm") is not None:
                p_entry_m = to_base_m(pose["p_entry_eye_mm"])
                entry = self._base_marker(22, Marker.SPHERE, ns="eye_debug")
                entry.pose.position.x = float(p_entry_m[0])
                entry.pose.position.y = float(p_entry_m[1])
                entry.pose.position.z = float(p_entry_m[2])
                entry.pose.orientation.w = 1.0
                entry.scale.x = 0.008
                entry.scale.y = 0.008
                entry.scale.z = 0.008
                entry.color.r = 1.0
                entry.color.g = 0.0
                entry.color.b = 0.0
                entry.color.a = 1.0
                markers.append(entry)

            # Text label near the eye center.
            text = self._base_marker(23, Marker.TEXT_VIEW_FACING, ns="eye_debug")
            text.pose.position.x = float(c_eye_m[0])
            text.pose.position.y = float(c_eye_m[1])
            text.pose.position.z = float(c_eye_m[2] + r_eye_m + 0.025)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.018
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"eye_debug\nr={1000.0 * r_eye_m:.1f} mm"
            markers.append(text)

            for _ in range(10):
                for marker in markers:
                    self.marker_pub.publish(marker)
                rclpy.spin_once(self, timeout_sec=0.05)

            self.get_logger().info(
                "RViz eye_debug markers published: "
                f"eye=({c_eye_m[0]:.3f}, {c_eye_m[1]:.3f}, {c_eye_m[2]:.3f}) m, "
                f"radius={r_eye_m:.3f} m"
            )


else:
    class TargetPoseVisualizer:
        def publish_target_pose(self, T_base_target: np.ndarray):
            return

        def publish_eye_and_axis(self, pose: dict, T_base_cam: np.ndarray):
            return


# ─────────────────────────────────────────────
# LOAD MODEL B
# ─────────────────────────────────────────────

def load_model_b(ckpt_path: Path) -> torch.nn.Module:
    """
    Load Model B:
        U-Net + ResNet34
        binary trocar segmentation
    """
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    try:
        ckpt = torch.load(
            ckpt_path,
            map_location=DEVICE,
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            ckpt_path,
            map_location=DEVICE,
        )

    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

    return model


# ─────────────────────────────────────────────
# LETTERBOX PREPROCESSING
# ─────────────────────────────────────────────

_norm_tf = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
    ToTensorV2(),
])


def letterbox_rgb(img: np.ndarray, img_size: int = 320):
    """
    Letterbox preprocessing for inference.

    This matches the logic used during training:
        1. resize while preserving aspect ratio
        2. pad to img_size x img_size

    Args:
        img:
            RGB crop, shape H x W x 3.

        img_size:
            Final square input size.

    Returns:
        padded:
            RGB image of shape img_size x img_size x 3.

        meta:
            Information required to invert the letterbox transform.
    """
    orig_h, orig_w = img.shape[:2]

    if orig_h <= 0 or orig_w <= 0:
        raise ValueError(f"Invalid image shape: {img.shape}")

    scale = img_size / max(orig_h, orig_w)

    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    new_w = max(1, min(img_size, new_w))
    new_h = max(1, min(img_size, new_h))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR,
    )

    pad_w = img_size - new_w
    pad_h = img_size - new_h

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    if padded.shape[:2] != (img_size, img_size):
        raise RuntimeError(
            f"Letterbox failed: got {padded.shape[:2]}, expected {(img_size, img_size)}"
        )

    meta = {
        "orig_h": orig_h,
        "orig_w": orig_w,
        "new_h": new_h,
        "new_w": new_w,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "scale": scale,
    }

    return padded, meta


def unletterbox_mask(mask_lbox: np.ndarray, meta: dict):
    """
    Convert predicted mask from letterbox coordinates back to original crop coordinates.

    Input:
        mask_lbox:
            Binary mask predicted on the 320x320 letterbox image.

    Output:
        mask_crop:
            Binary mask with the same H x W as the original rgb_crop / xyz_crop.
    """
    orig_h = meta["orig_h"]
    orig_w = meta["orig_w"]

    new_h = meta["new_h"]
    new_w = meta["new_w"]

    pad_top = meta["pad_top"]
    pad_left = meta["pad_left"]

    # 1. Remove padding.
    mask_no_pad = mask_lbox[
        pad_top:pad_top + new_h,
        pad_left:pad_left + new_w,
    ]

    # 2. Resize back to original crop size.
    mask_crop = cv2.resize(
        mask_no_pad.astype(np.uint8),
        (orig_w, orig_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask_crop.astype(np.uint8)


# ─────────────────────────────────────────────
# MODEL B INFERENCE
# ─────────────────────────────────────────────

def predict_trocar_bbox(model_b: torch.nn.Module, rgb_crop: np.ndarray):
    """
    Predict trocar mask and bbox inside the original eye crop.

    Important:
        The model receives a 320x320 letterbox image.
        The output mask is then unletterboxed back to the original crop size.

    Args:
        model_b:
            U-Net model.

        rgb_crop:
            Eye crop from the original RGB image, H x W x 3.

    Returns:
        bbox_local:
            [x1, y1, x2, y2] in original crop coordinates,
            or None if no trocar pixels are detected.

        mask_crop:
            Binary trocar mask in original crop coordinates.
            Shape is exactly rgb_crop.shape[:2].
    """
    crop_h, crop_w = rgb_crop.shape[:2]

    # 1. Letterbox crop to 320x320.
    rgb_lbox, meta = letterbox_rgb(
        img=rgb_crop,
        img_size=IMG_SIZE,
    )

    # 2. Normalize and convert to tensor.
    inp = _norm_tf(image=rgb_lbox)["image"]
    inp = inp.unsqueeze(0).to(DEVICE)

    # 3. Model prediction in letterbox coordinates.
    with torch.no_grad():
        logits = model_b(inp)
        prob = torch.sigmoid(logits)
        mask_lbox = (prob > SEG_THRESH)
        mask_lbox = mask_lbox.squeeze().cpu().numpy().astype(np.uint8)

    # 4. Convert mask back to original crop coordinates.
    mask_crop = unletterbox_mask(
        mask_lbox=mask_lbox,
        meta=meta,
    )

    if mask_crop.shape[:2] != (crop_h, crop_w):
        raise RuntimeError(
            f"Mask/crop size mismatch: mask={mask_crop.shape[:2]}, crop={(crop_h, crop_w)}"
        )

    # 5. Get trocar bbox in original crop coordinates.
    ys, xs = np.where(mask_crop > 0)

    if len(xs) == 0:
        return None, mask_crop

    bbox_local = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]

    return bbox_local, mask_crop


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(rgb, xyz, model_A, model_B):
    """
    Full inference pipeline.

    Args:
        rgb:
            Full RGB image, H x W x 3.

        xyz:
            Full organized point cloud, H x W x 3.

        model_A:
            YOLO eye detector.

        model_B:
            U-Net trocar segmentation model.

    Returns:
        Dictionary with detection, segmentation, 3D, pose results and timings,
        or None if the eye is not detected.
    """
    timings_ms = {
        "eye_detection_ms": None,
        "eye_crop_generation_ms": None,
        "trocar_segmentation_ms": None,
        "pose_estimation_ms": None,
        "planning_time_ms": None,
        "total_pipeline_time_ms": None,
    }

    # ── Model A: detect eye in full image ─────────────────────
    t_step = time.perf_counter()
    sync_cuda_if_needed()
    res_A = model_A(rgb, verbose=False)[0]
    sync_cuda_if_needed()
    timings_ms["eye_detection_ms"] = elapsed_ms(t_step)

    boxes_A = [
        b for b in res_A.boxes
        if float(b.conf) >= CONF_A
    ]

    if not boxes_A:
        print(f"  [A] Eye not detected ({timings_ms['eye_detection_ms']:.1f} ms)")
        return None

    best_A = max(boxes_A, key=lambda b: float(b.conf))

    eye_bbox = best_A.xyxy[0].cpu().numpy().astype(int).tolist()
    conf_A = float(best_A.conf)

    # ── Crop RGB + XYZ using the same crop window ─────────────
    t_step = time.perf_counter()
    rgb_crop, xyz_crop, offset = crop_rgb_and_xyz(
        rgb=rgb,
        xyz=xyz,
        eye_bbox=eye_bbox,
        margin=MARGIN,
    )
    timings_ms["eye_crop_generation_ms"] = elapsed_ms(t_step)

    if xyz_crop is None:
        print("  [XYZ] XYZ crop unavailable")
        return {
            "eye_bbox": eye_bbox,
            "conf_A": round(conf_A, 3),
            "trocar_global": None,
            "rgb_crop": rgb_crop,
            "xyz_crop": xyz_crop,
            "seg_mask": None,
            "offset": offset,
            "trocar_3d_mm": None,
            "trocar_pointcloud": None,
            "pose": None,
            "timings_ms": timings_ms,
        }

    # ── Model B: U-Net segmentation with letterbox ────────────
    t_step = time.perf_counter()
    sync_cuda_if_needed()
    trocar_local, seg_mask = predict_trocar_bbox(
        model_b=model_B,
        rgb_crop=rgb_crop,
    )
    sync_cuda_if_needed()
    timings_ms["trocar_segmentation_ms"] = elapsed_ms(t_step)

    # Empty mask = trocar not detected.
    if trocar_local is None:
        print("  [B] Trocar not detected in crop (empty mask)")
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

    # ── Local crop coordinates → global image coordinates ─────
    trocar_global = bbox_local_to_global(
        bbox=trocar_local,
        offset=offset,
    )

    # ── 3D position from predicted mask ───────────────────────
    # seg_mask is already back in original crop coordinates,
    # so it is pixel-aligned with xyz_crop.
    trocar_3d, trocar_pointcloud = get_3d_from_mask(
        xyz_crop=xyz_crop,
        seg_mask=seg_mask,
    )

    # ── Pose estimation ───────────────────────────────────────
    pose_result = None

    if trocar_pointcloud is not None:
        t_step = time.perf_counter()
        pose_result = estimate_trocar_pose(
            xyz_crop=xyz_crop,
            trocar_mask=seg_mask,
            trocar_points=trocar_pointcloud,
        )
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


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

def draw_results(rgb_bgr, result):
    """
    Draw eye bbox, trocar bbox and 3D point on the image.

    Args:
        rgb_bgr:
            Image in BGR format, ready for OpenCV saving.

        result:
            Output dictionary from run_pipeline().
    """
    vis = rgb_bgr.copy()

    if result is None:
        return vis

    # Eye bbox.
    x1, y1, x2, y2 = result["eye_bbox"]

    cv2.rectangle(
        vis,
        (x1, y1),
        (x2, y2),
        (255, 100, 0),
        2,
    )

    cv2.putText(
        vis,
        f"eye {result['conf_A']:.2f}",
        (x1, y1 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 100, 0),
        1,
    )

    # Trocar bbox.
    if result["trocar_global"] is not None:
        tx1, ty1, tx2, ty2 = result["trocar_global"]

        cv2.rectangle(
            vis,
            (tx1, ty1),
            (tx2, ty2),
            (0, 220, 0),
            2,
        )

        cv2.putText(
            vis,
            "trocar",
            (tx1, ty1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 220, 0),
            1,
        )

        if result["trocar_3d_mm"] is not None:
            x3d, y3d, z3d = result["trocar_3d_mm"]

            label = f"X{x3d:.1f} Y{y3d:.1f} Z{z3d:.1f} mm"

            cv2.putText(
                vis,
                label,
                (tx1, ty2 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 0),
                1,
            )

    else:
        cv2.putText(
            vis,
            "trocar not detected",
            (x1, y2 + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 100, 255),
            1,
        )

    return vis

# ─────────────────────────────────────────────
# 3D DEBUG OUTPUTS FOR EVALUATION
# ─────────────────────────────────────────────

def save_rgb_image(path: Path, rgb_img: np.ndarray):
    """
    Save an RGB image using OpenCV.
    """
    if rgb_img is None:
        return
    cv2.imwrite(str(path), cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))


def save_mask_overlay(rgb_crop, mask_crop, out_path):
    """
    Save an RGB crop with the predicted trocar mask overlaid in green.
    """
    if rgb_crop is None or mask_crop is None:
        return

    vis = rgb_crop.copy()
    overlay = vis.copy()
    overlay[mask_crop > 0] = [0, 255, 0]
    blended = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    save_rgb_image(out_path, blended)


def sample_points(points, max_points=5000):
    if points is None:
        return None

    points = np.asarray(points)

    if len(points) == 0:
        return points

    if len(points) <= max_points:
        return points

    idx = np.random.choice(len(points), max_points, replace=False)
    return points[idx]


def get_eye_points_for_visualization(xyz_crop, trocar_mask, trocar_points):
    """
    Reconstruct a light eye point cloud only for visualization.

    The pose itself is estimated by pose_utils. This helper is used only if
    pose_utils did not return the exact eye_points_used array.
    """
    if xyz_crop is None or trocar_mask is None or trocar_points is None:
        return None

    if len(trocar_points) == 0:
        return None

    z_trocar_approx = np.median(trocar_points[:, 2])

    eye_points = xyz_crop[trocar_mask == 0]

    finite = np.isfinite(eye_points).all(axis=1)
    eye_points = eye_points[finite]

    nonzero = np.linalg.norm(eye_points, axis=1) > 1e-6
    eye_points = eye_points[nonzero]

    eye_points = eye_points[np.abs(eye_points[:, 2] - z_trocar_approx) < 20.0]

    return eye_points


def R_x_180():
    """
    Rotation +180 degrees around X, only for the Matplotlib debug view.
    It makes the plot easier to interpret visually; it does not modify the
    saved camera-frame PLY files.
    """
    return np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=np.float64)


def transform_points_for_viz(points):
    if points is None:
        return None

    R = R_x_180()
    points = np.asarray(points, dtype=np.float64)

    return (R @ points.T).T


def transform_vector_for_viz(v):
    R = R_x_180()
    v = np.asarray(v, dtype=np.float64)
    v = R @ v

    norm = np.linalg.norm(v)
    if norm < 1e-9:
        return v

    return v / norm


def get_pose_p_pre(pose):
    """
    Return the pre-entry / target point using the new key p_pre_mm.

    A fallback to p_target_mm is kept so that the inference script also works
    with older pose_utils outputs. The JSON outputs only expose p_pre_mm.
    """
    if pose is None:
        return None

    if pose.get("p_pre_mm") is not None:
        return np.asarray(pose["p_pre_mm"], dtype=np.float64)

    if pose.get("p_target_mm") is not None:
        return np.asarray(pose["p_target_mm"], dtype=np.float64)

    return None


def plot_pose_3d(eye_points, trocar_points, pose, out_path):
    """
    Save a Matplotlib 3D debug plot with:
      - eye points
      - trocar points
      - fitted eye sphere
      - trocar axis
      - eye center
      - p_entry_eye
      - p_entry_trocar
      - p_pre
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    eye_points_viz = transform_points_for_viz(eye_points)
    trocar_points_viz = transform_points_for_viz(trocar_points)

    p0 = transform_points_for_viz(np.asarray(pose["p0_trocar_mm"])[None, :])[0]
    c_eye = transform_points_for_viz(np.asarray(pose["eye_center_mm"])[None, :])[0]
    p_entry_eye = transform_points_for_viz(np.asarray(pose["p_entry_eye_mm"])[None, :])[0]
    p_entry_trocar = transform_points_for_viz(np.asarray(pose["p_entry_trocar_mm"])[None, :])[0]

    p_pre_raw = get_pose_p_pre(pose)
    if p_pre_raw is None:
        raise KeyError("pose does not contain p_pre_mm or p_target_mm")
    p_pre = transform_points_for_viz(p_pre_raw[None, :])[0]

    v = transform_vector_for_viz(pose["v_trocar"])
    r_eye = float(pose["eye_radius_mm"])

    eye_vis = sample_points(eye_points_viz, max_points=4000)
    trocar_vis = sample_points(trocar_points_viz, max_points=1000)

    if eye_vis is not None and len(eye_vis) > 0:
        ax.scatter(
            eye_vis[:, 0], eye_vis[:, 1], eye_vis[:, 2],
            s=1, alpha=0.3, label="eye points"
        )

    if trocar_vis is not None and len(trocar_vis) > 0:
        ax.scatter(
            trocar_vis[:, 0], trocar_vis[:, 1], trocar_vis[:, 2],
            s=8, alpha=0.9, label="trocar points"
        )

    ax.scatter([c_eye[0]], [c_eye[1]], [c_eye[2]], s=80, label="eye center")
    ax.scatter([p_entry_eye[0]], [p_entry_eye[1]], [p_entry_eye[2]], s=100, label="p_entry_eye")
    ax.scatter([p_entry_trocar[0]], [p_entry_trocar[1]], [p_entry_trocar[2]], s=100, label="p_entry_trocar")
    ax.scatter([p_pre[0]], [p_pre[1]], [p_pre[2]], s=100, label="p_pre")
    ax.scatter([p0[0]], [p0[1]], [p0[2]], s=60, label="trocar axis point p0")

    t = np.linspace(-60, 30, 150)
    line = p0[None, :] + t[:, None] * v[None, :]
    ax.plot(line[:, 0], line[:, 1], line[:, 2], linewidth=2, label="trocar axis")

    u = np.linspace(0, 2 * np.pi, 30)
    vv = np.linspace(0, np.pi, 20)

    xs = c_eye[0] + r_eye * np.outer(np.cos(u), np.sin(vv))
    ys = c_eye[1] + r_eye * np.outer(np.sin(u), np.sin(vv))
    zs = c_eye[2] + r_eye * np.outer(np.ones_like(u), np.cos(vv))

    ax.plot_wireframe(xs, ys, zs, rstride=2, cstride=2, alpha=0.2)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.set_title("Pose estimation debug - visual frame Rx(180)")
    ax.legend()

    all_pts = []
    if eye_vis is not None and len(eye_vis) > 0:
        all_pts.append(eye_vis)
    if trocar_vis is not None and len(trocar_vis) > 0:
        all_pts.append(trocar_vis)

    all_pts.append(np.array([c_eye, p_entry_eye, p_entry_trocar, p_pre, p0]))
    all_pts = np.vstack(all_pts)

    mins = all_pts.min(axis=0)
    maxs = all_pts.max(axis=0)
    centers = (mins + maxs) / 2
    radius = max((maxs - mins).max() / 2, 1.0)

    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=200)
    plt.close(fig)

def find_nearest_pixel_in_xyz(xyz_crop, point_3d):
    if xyz_crop is None or point_3d is None:
        return None

    valid = np.isfinite(xyz_crop).all(axis=2)
    valid &= np.linalg.norm(xyz_crop, axis=2) > 1e-6

    coords = np.argwhere(valid)   # (N, 2) = [y, x]
    pts = xyz_crop[valid]         # (N, 3)

    if len(pts) == 0:
        return None

    point_3d = np.asarray(point_3d, dtype=np.float64)
    dists = np.linalg.norm(pts - point_3d, axis=1)
    idx = np.argmin(dists)

    y, x = coords[idx]
    return int(x), int(y), float(dists[idx])


def compute_pose_quality_checks(pose):
    """
    Basic checks to validate the estimated pose geometry.
    """
    p0 = np.asarray(pose["p0_trocar_mm"], dtype=np.float64)
    v = np.asarray(pose["v_trocar"], dtype=np.float64)
    p_entry_eye = np.asarray(pose["p_entry_eye_mm"], dtype=np.float64)
    p_entry_trocar = np.asarray(pose["p_entry_trocar_mm"], dtype=np.float64)
    p_pre = get_pose_p_pre(pose)
    c_eye = np.asarray(pose["eye_center_mm"], dtype=np.float64)

    if p_pre is None:
        raise KeyError("pose does not contain p_pre_mm or p_target_mm")

    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        raise ValueError("v_trocar has near-zero norm")

    v = v / norm_v

    t_entry_eye_mm = float(np.dot(p_entry_eye - p0, v))
    t_entry_trocar_mm = float(np.dot(p_entry_trocar - p0, v))
    t_pre_mm = float(np.dot(p_pre - p0, v))
    dot_axis_to_eye = float(np.dot(v, c_eye - p0))
    axis_points_towards_eye = dot_axis_to_eye > 0

    return {
        "t_entry_eye_mm": float(t_entry_eye_mm),
        "t_entry_trocar_mm": float(t_entry_trocar_mm),
        "t_pre_mm": float(t_pre_mm),
        "dot_axis_to_eye": float(dot_axis_to_eye),
        "axis_points_towards_eye": bool(axis_points_towards_eye),
    }

def compute_trocar_axis_fit_quality(trocar_points, pose):
    """
    Compute perpendicular distances from segmented trocar points to the estimated axis.
    """
    points = np.asarray(trocar_points, dtype=np.float64)

    p0 = np.asarray(pose["p0_trocar_mm"], dtype=np.float64)
    v = np.asarray(pose["v_trocar"], dtype=np.float64)

    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        raise ValueError("v_trocar has near-zero norm")

    v = v / norm_v

    diffs = points - p0[None, :]
    dists = np.linalg.norm(np.cross(diffs, v[None, :]), axis=1)

    return {
        "trocar_axis_dist_median_mm": float(np.median(dists)),
        "trocar_axis_dist_p90_mm": float(np.percentile(dists, 90)),
    }


def save_pose_quality_checks(pose_checks, axis_fit_checks, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("POSE QUALITY CHECKS\n")
        f.write("===================\n\n")

        f.write("Basic pose direction checks\n")
        f.write("---------------------------\n")
        for key, value in pose_checks.items():
            f.write(f"{key}: {value}\n")

        f.write("\nTrocar axis fit checks\n")
        f.write("----------------------\n")
        for key, value in axis_fit_checks.items():
            f.write(f"{key}: {value}\n")

        f.write("\nINTERPRETATION\n")
        f.write("--------------\n")
        f.write("t_entry_mm should usually be positive if v_trocar points from p0 towards the eye.\n")
        f.write("axis_points_towards_eye should be True.\n")
        f.write("trocar_axis_dist_median_mm is the typical perpendicular distance from trocar points to the estimated axis.\n")
        f.write("trocar_axis_dist_p90_mm indicates whether many trocar points are far from the estimated axis.\n")


def get_T_cam_target_from_pose(pose):
    """
    Return the final target pose computed by pose_utils.

    The clearance is no longer computed in this inference script.
    pose_utils.py already returns T_cam_target using:
        p_entry_eye_mm -> p_entry_trocar_mm -> p_pre_mm
    """
    if pose is None or pose.get("T_cam_target") is None:
        raise ValueError("pose does not contain T_cam_target")

    return np.asarray(pose["T_cam_target"], dtype=np.float64)

def draw_pose_on_crop(rgb_crop, xyz_crop, trocar_3d, pose, out_path):
    """
    Save the crop with the estimated trocar centroid and entry points projected
    to the nearest valid XYZ pixels.

    """
    if rgb_crop is None or xyz_crop is None or pose is None:
        return

    vis = rgb_crop.copy()

    centroid_px = find_nearest_pixel_in_xyz(xyz_crop, trocar_3d)
    if centroid_px is not None:
        x, y, _ = centroid_px
        cv2.circle(vis, (x, y), 4, (255, 255, 0), -1)  # yellow in RGB
        cv2.putText(
            vis,
            "trocar_3d",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
        )

    p_entry_eye = pose["p_entry_eye_mm"]
    entry_eye_px = find_nearest_pixel_in_xyz(xyz_crop, p_entry_eye)
    if entry_eye_px is not None:
        x, y, _ = entry_eye_px
        cv2.circle(vis, (x, y), 5, (255, 0, 0), -1)  # red in RGB
        cv2.putText(
            vis,
            "p_entry_eye",
            (x + 6, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
        )

    p_entry_trocar = pose["p_entry_trocar_mm"]
    entry_trocar_px = find_nearest_pixel_in_xyz(xyz_crop, p_entry_trocar)
    if entry_trocar_px is not None:
        x, y, _ = entry_trocar_px
        cv2.circle(vis, (x, y), 5, (0, 255, 255), -1)  # cyan in RGB
        cv2.putText(
            vis,
            "p_entry_trocar",
            (x + 6, y + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

    save_rgb_image(out_path, vis)

def write_colored_ply(path, points, colors=None):
    """
    Save an ASCII PLY point cloud with RGB colors.
    """
    points = np.asarray(points, dtype=np.float32)

    if colors is None:
        colors = np.tile(
            np.array([[255, 255, 255]], dtype=np.uint8),
            (len(points), 1),
        )
    else:
        colors = np.asarray(colors, dtype=np.uint8)

    assert len(points) == len(colors)

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for p, c in zip(points, colors):
            f.write(
                f"{p[0]} {p[1]} {p[2]} "
                f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
            )


def write_rgb_xyz_ply(path, xyz, rgb):
    """
    Save the captured organized RGB-D point cloud as a colored PLY.

    This exports the full valid cloud from the Zivid capture:
        - XYZ coordinates in camera frame, in millimetres.
        - RGB colors from the captured RGB image.

    Invalid points, NaNs/Infs and all-zero XYZ points are removed.
    """
    if xyz is None or rgb is None:
        return

    if xyz.shape[:2] != rgb.shape[:2]:
        raise ValueError(
            f"RGB/XYZ size mismatch for PLY export: "
            f"rgb={rgb.shape[:2]}, xyz={xyz.shape[:2]}"
        )

    xyz_flat = xyz.reshape(-1, 3)
    rgb_flat = rgb.reshape(-1, 3)

    valid = np.isfinite(xyz_flat).all(axis=1)
    valid &= np.linalg.norm(xyz_flat, axis=1) > 1e-6

    points = xyz_flat[valid].astype(np.float32)
    colors = rgb_flat[valid].astype(np.uint8)

    write_colored_ply(path, points, colors)


def create_sphere_points(center, radius, n_u=80, n_v=40):
    center = np.asarray(center, dtype=np.float64)
    radius = float(radius)

    u = np.linspace(0, 2 * np.pi, n_u)
    v = np.linspace(0, np.pi, n_v)

    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    return np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()]).astype(np.float32)


def create_pose_debug_ply(eye_points, trocar_points, pose, out_path):
    """
    Export one combined CloudCompare PLY:
      - eye points: light blue
      - trocar points: green
      - trocar axis: magenta
      - p_entry_eye: red
      - p_entry_trocar: cyan
      - p_pre: blue
      - eye center: orange
      - fitted sphere: pale yellow

    This function is kept for optional manual debugging, but it is no longer
    called by the main pipeline.
    """
    p0 = np.asarray(pose["p0_trocar_mm"], dtype=np.float64)
    v = np.asarray(pose["v_trocar"], dtype=np.float64)
    c_eye = np.asarray(pose["eye_center_mm"], dtype=np.float64)
    r_eye = float(pose["eye_radius_mm"])
    p_entry_eye = np.asarray(pose["p_entry_eye_mm"], dtype=np.float64)
    p_entry_trocar = np.asarray(pose["p_entry_trocar_mm"], dtype=np.float64)
    p_pre = get_pose_p_pre(pose)

    if p_pre is None:
        raise KeyError("pose does not contain p_pre_mm or p_target_mm")

    eye_vis = sample_points(eye_points, max_points=8000)
    trocar_vis = sample_points(trocar_points, max_points=2000)

    pts_all = []
    cols_all = []

    if eye_vis is not None and len(eye_vis) > 0:
        pts_all.append(eye_vis)
        cols_all.append(
            np.tile(np.array([[120, 180, 255]], dtype=np.uint8), (len(eye_vis), 1))
        )

    if trocar_vis is not None and len(trocar_vis) > 0:
        pts_all.append(trocar_vis)
        cols_all.append(
            np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(trocar_vis), 1))
        )

    t = np.linspace(-60, 30, 300)
    axis_pts = p0[None, :] + t[:, None] * v[None, :]
    pts_all.append(axis_pts)
    cols_all.append(np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (len(axis_pts), 1)))

    entry_eye_pts = np.tile(p_entry_eye[None, :], (50, 1))
    pts_all.append(entry_eye_pts)
    cols_all.append(np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(entry_eye_pts), 1)))

    entry_trocar_pts = np.tile(p_entry_trocar[None, :], (50, 1))
    pts_all.append(entry_trocar_pts)
    cols_all.append(np.tile(np.array([[0, 255, 255]], dtype=np.uint8), (len(entry_trocar_pts), 1)))

    pre_pts = np.tile(p_pre[None, :], (50, 1))
    pts_all.append(pre_pts)
    cols_all.append(np.tile(np.array([[0, 80, 255]], dtype=np.uint8), (len(pre_pts), 1)))

    center_pts = np.tile(c_eye[None, :], (50, 1))
    pts_all.append(center_pts)
    cols_all.append(np.tile(np.array([[255, 165, 0]], dtype=np.uint8), (len(center_pts), 1)))

    sphere_pts = create_sphere_points(c_eye, r_eye, n_u=40, n_v=20)
    pts_all.append(sphere_pts)
    cols_all.append(np.tile(np.array([[255, 220, 120]], dtype=np.uint8), (len(sphere_pts), 1)))

    points = np.vstack(pts_all).astype(np.float32)
    colors = np.vstack(cols_all).astype(np.uint8)

    write_colored_ply(out_path, points, colors)

def create_sphere_axis_debug_ply(pose, out_path):
    """
    Export a minimal CloudCompare PLY with only:
      - fitted eye sphere: yellow
      - estimated insertion axis: magenta
    """
    c_eye = np.asarray(pose["eye_center_mm"], dtype=np.float64)
    r_eye = float(pose["eye_radius_mm"])
    p0 = np.asarray(pose["p0_trocar_mm"], dtype=np.float64)
    v_axis = np.asarray(pose["v_trocar"], dtype=np.float64)

    norm_v = np.linalg.norm(v_axis)
    if norm_v < 1e-9:
        raise ValueError("v_trocar has near-zero norm")
    v_axis = v_axis / norm_v

    sphere_points = create_sphere_points(c_eye, r_eye, n_u=80, n_v=40)
    sphere_colors = np.tile(
        np.array([[255, 220, 0]], dtype=np.uint8),
        (len(sphere_points), 1),
    )

    t = np.linspace(-60, 30, 500)
    axis_points = (p0[None, :] + t[:, None] * v_axis[None, :]).astype(np.float32)
    axis_colors = np.tile(
        np.array([[255, 0, 255]], dtype=np.uint8),
        (len(axis_points), 1),
    )

    points = np.vstack([sphere_points, axis_points]).astype(np.float32)
    colors = np.vstack([sphere_colors, axis_colors]).astype(np.uint8)

    write_colored_ply(out_path, points, colors)


def get_eye_points_used_for_sphere(xyz_crop, trocar_mask, trocar_points, pose=None):
    """
    Return eye points before and after filtering.

    If pose_utils returned eye_points_used/sphere_inliers, use those because
    they match the actual pose estimation path.
    """
    eye_raw = None

    if xyz_crop is not None and trocar_mask is not None and trocar_points is not None and len(trocar_points) > 0:
        z_trocar_approx = np.median(trocar_points[:, 2])
        eye_points = xyz_crop[trocar_mask == 0]

        finite = np.isfinite(eye_points).all(axis=1)
        eye_points = eye_points[finite]

        nonzero = np.linalg.norm(eye_points, axis=1) > 1e-6
        eye_points = eye_points[nonzero]

        eye_raw = eye_points[np.abs(eye_points[:, 2] - z_trocar_approx) < 20.0]

    eye_used = None
    sphere_inliers = None

    if pose is not None:
        if pose.get("eye_points_used") is not None:
            eye_used = np.asarray(pose["eye_points_used"], dtype=np.float32)
        if pose.get("sphere_inliers") is not None:
            sphere_inliers = np.asarray(pose["sphere_inliers"], dtype=np.float32)

    if eye_used is None and eye_raw is not None:
        eye_used = filter_points(eye_raw)

    return eye_raw, eye_used, sphere_inliers


def export_eye_sphere_debug_plys(xyz_crop, trocar_mask, trocar_points, pose, out_dir):
    """
    Export separate PLYs to inspect in CloudCompare.
    """
    eye_raw, eye_used, sphere_inliers = get_eye_points_used_for_sphere(
        xyz_crop=xyz_crop,
        trocar_mask=trocar_mask,
        trocar_points=trocar_points,
        pose=pose,
    )

    if eye_used is not None and len(eye_used) > 0:
        colors_used = np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (len(eye_used), 1))
        write_colored_ply(out_dir / "debug_eye_points_USED_FOR_SPHERE.ply", eye_used, colors_used)

    c_eye = np.asarray(pose["eye_center_mm"], dtype=np.float64)
    r_eye = float(pose["eye_radius_mm"])
    p_entry_eye = np.asarray(pose["p_entry_eye_mm"], dtype=np.float64)
    p_entry_trocar = np.asarray(pose["p_entry_trocar_mm"], dtype=np.float64)
    p_pre = get_pose_p_pre(pose)
    p0 = np.asarray(pose["p0_trocar_mm"], dtype=np.float64)
    v_axis = np.asarray(pose["v_trocar"], dtype=np.float64)

    if p_pre is None:
        raise KeyError("pose does not contain p_pre_mm or p_target_mm")

    sphere_points = create_sphere_points(c_eye, r_eye)
    sphere_colors = np.tile(np.array([[255, 220, 0]], dtype=np.uint8), (len(sphere_points), 1))
    write_colored_ply(out_dir / "debug_fitted_sphere_points.ply", sphere_points, sphere_colors)

    t = np.linspace(-60, 30, 300)
    axis_points = p0[None, :] + t[:, None] * v_axis[None, :]

    entry_eye_points = np.tile(p_entry_eye[None, :], (100, 1))
    entry_trocar_points = np.tile(p_entry_trocar[None, :], (100, 1))
    pre_points = np.tile(p_pre[None, :], (100, 1))
    center_points = np.tile(c_eye[None, :], (100, 1))

    all_points = []
    all_colors = []

    if eye_used is not None and len(eye_used) > 0:
        all_points.append(eye_used)
        all_colors.append(np.tile(np.array([[255, 255, 255]], dtype=np.uint8), (len(eye_used), 1)))

    all_points.append(trocar_points)
    all_colors.append(np.tile(np.array([[0, 255, 0]], dtype=np.uint8), (len(trocar_points), 1)))

    all_points.append(sphere_points)
    all_colors.append(sphere_colors)

    all_points.append(axis_points)
    all_colors.append(np.tile(np.array([[255, 0, 255]], dtype=np.uint8), (len(axis_points), 1)))

    all_points.append(entry_eye_points)
    all_colors.append(np.tile(np.array([[255, 0, 0]], dtype=np.uint8), (len(entry_eye_points), 1)))

    all_points.append(entry_trocar_points)
    all_colors.append(np.tile(np.array([[0, 255, 255]], dtype=np.uint8), (len(entry_trocar_points), 1)))

    all_points.append(pre_points)
    all_colors.append(np.tile(np.array([[0, 80, 255]], dtype=np.uint8), (len(pre_points), 1)))

    all_points.append(center_points)
    all_colors.append(np.tile(np.array([[255, 165, 0]], dtype=np.uint8), (len(center_points), 1)))

    all_points = np.vstack(all_points).astype(np.float32)
    all_colors = np.vstack(all_colors).astype(np.uint8)

    write_colored_ply(out_dir / "debug_COMBINED_eye_sphere_pose.ply", all_points, all_colors)

def make_result_summary(result):
    """
    Small JSON summary without dumping large point clouds.
    """
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
        if p_pre is None:
            raise KeyError("pose does not contain p_pre_mm or p_target_mm")

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
        }

        if pose.get("T_cam_target") is not None:
            summary["pose"]["T_cam_target"] = np.asarray(pose["T_cam_target"]).tolist()

        if pose.get("T_base_target") is not None:
            summary["pose"]["T_base_target"] = np.asarray(pose["T_base_target"]).tolist()

    return summary


def make_prediction_json(result):
    """
    Minimal prediction JSON used by gt_evaluate.py.

    This keeps the evaluation contract separate from the larger debug summary.
    """
    if result is None or result.get("pose") is None:
        prediction = {
            "status": "no_pose",
            "pose": None,
        }
        if result is not None and result.get("timings_ms") is not None:
            prediction["timings_ms"] = result.get("timings_ms")
        return prediction

    pose = result["pose"]
    p_pre = get_pose_p_pre(pose)
    if p_pre is None:
        raise KeyError("pose does not contain p_pre_mm or p_target_mm")

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

            # Required/recommended for GT annotation projection.
            "eye_center_mm": np.asarray(pose["eye_center_mm"]).tolist(),
            "eye_radius_mm": float(pose["eye_radius_mm"]),

            # Pose/eye diagnostics.
            "p_visible_eye_mm": np.asarray(pose["p_visible_eye_mm"]).tolist()
            if pose.get("p_visible_eye_mm") is not None else None,
            "n_anterior": np.asarray(pose["n_anterior"]).tolist()
            if pose.get("n_anterior") is not None else None,
            "p_posterior_pole_mm": np.asarray(pose["p_posterior_pole_mm"]).tolist()
            if pose.get("p_posterior_pole_mm") is not None else None,
            "p0_trocar_mm": np.asarray(pose["p0_trocar_mm"]).tolist()
            if pose.get("p0_trocar_mm") is not None else None,
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

    # Flat duplicates keep this file aligned with the GT-mask batch prediction.
    prediction.update(prediction["pose"])

    return prediction


def save_evaluation_prediction(result, out_dir):
    """
    Save the compact prediction file consumed by gt_evaluate.py.
    """
    with open(out_dir / "prediction.json", "w", encoding="utf-8") as f:
        json.dump(make_prediction_json(result), f, indent=2)


def save_debug_outputs(nombre, rgb, xyz, result, out_dir):
    """
    Save all debug outputs for one final capture attempt.

    These outputs are equivalent to prueba.py, but generated from the real
    inference results: YOLO eye bbox + U-Net predicted trocar mask.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    save_rgb_image(out_dir / f"{nombre}_rgb.png", rgb)

    if xyz is not None:
        np.save(out_dir / f"{nombre}_xyz.npy", xyz.astype(np.float32))
        write_rgb_xyz_ply(out_dir / f"{nombre}_captured_cloud.ply", xyz, rgb)

    with open(out_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_result_summary(result), f, indent=2)
    save_evaluation_prediction(result, out_dir)

    if result is None:
        return

    eye_bbox = result.get("eye_bbox")
    if rgb is not None and eye_bbox is not None:
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

        if rgb_crop is not None:
            save_mask_overlay(
                rgb_crop=rgb_crop,
                mask_crop=seg_mask,
                out_path=out_dir / "debug_04_eye_crop_plus_trocar_mask.png",
            )

    if pose is None or trocar_points is None or xyz_crop is None or seg_mask is None:
        return

    eye_points = None
    if pose.get("eye_points_used") is not None:
        eye_points = np.asarray(pose["eye_points_used"], dtype=np.float32)
    else:
        eye_points = get_eye_points_for_visualization(
            xyz_crop=xyz_crop,
            trocar_mask=seg_mask,
            trocar_points=trocar_points,
        )

    draw_pose_on_crop(
        rgb_crop=rgb_crop,
        xyz_crop=xyz_crop,
        trocar_3d=trocar_3d,
        pose=pose,
        out_path=out_dir / "debug_05_pose_on_crop.png",
    )

    plot_pose_3d(
        eye_points=eye_points,
        trocar_points=trocar_points,
        pose=pose,
        out_path=out_dir / "debug_06_pose_3d.png",
    )

    create_sphere_axis_debug_ply(
        pose=pose,
        out_path=out_dir / "debug.ply",
    )

    export_eye_sphere_debug_plys(
        xyz_crop=xyz_crop,
        trocar_mask=seg_mask,
        trocar_points=trocar_points,
        pose=pose,
        out_dir=out_dir,
    )

    if pose.get("T_cam_target") is not None:
        np.save(
            out_dir / "T_cam_target.npy",
            np.asarray(pose["T_cam_target"], dtype=np.float64),
        )

    if pose.get("T_base_target") is not None:
        np.save(
            out_dir / "T_base_target.npy",
            np.asarray(pose["T_base_target"], dtype=np.float64),
        )

    # Re-write summary after all pose information has been added.
    with open(out_dir / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(make_result_summary(result), f, indent=2)
    save_evaluation_prediction(result, out_dir)

# ─────────────────────────────────────────────
# ROBOT TARGET GENERATION
# ─────────────────────────────────────────────
def get_meca_pose_script_path() -> Path:
    """Return the second-stage robot-pose script path."""
    if MECA_POSE_SCRIPT.exists():
        return MECA_POSE_SCRIPT

    # Fallback: same directory as this pipeline script.
    candidate = Path(__file__).resolve().parent / MECA_POSE_SCRIPT.name
    if candidate.exists():
        return candidate

    return MECA_POSE_SCRIPT


def get_movelin_advance_script_path() -> Path:
    """Return the MoveLin advance helper script path."""
    if MOVELIN_ADVANCE_SCRIPT.exists():
        return MOVELIN_ADVANCE_SCRIPT

    candidate = Path(__file__).resolve().parent / MOVELIN_ADVANCE_SCRIPT.name
    if candidate.exists():
        return candidate

    return MOVELIN_ADVANCE_SCRIPT


def get_native_movepose_script_path() -> Path:
    """Return the native Meca500 MovePose helper script path."""
    if NATIVE_MOVEPOSE_SCRIPT.exists():
        return NATIVE_MOVEPOSE_SCRIPT

    candidate = Path(__file__).resolve().parent / NATIVE_MOVEPOSE_SCRIPT.name
    if candidate.exists():
        return candidate

    return NATIVE_MOVEPOSE_SCRIPT


def get_j6_rotate_script_path() -> Path:
    """Return the interactive J6 rotation helper script path."""
    if J6_ROTATE_SCRIPT.exists():
        return J6_ROTATE_SCRIPT

    candidate = Path(__file__).resolve().parent / J6_ROTATE_SCRIPT.name
    if candidate.exists():
        return candidate

    return J6_ROTATE_SCRIPT


def make_stage_matrix(T_base_target: np.ndarray, extra_clearance_mm: float) -> np.ndarray:
    """
    Build T_stage from T_base_target by moving backwards along -Z of T_base_target.

    T_base_target is the final native target at 50 mm clearance. T_stage is only
    used for MoveIt staging; final precision is handled by native MovePose.
    """
    T_stage = np.asarray(T_base_target, dtype=np.float64).copy()

    if abs(float(extra_clearance_mm)) < 1e-12:
        return T_stage

    z_ins = T_stage[:3, 2].astype(np.float64)
    z_norm = np.linalg.norm(z_ins)
    if z_norm < 1e-9:
        raise ValueError("T_base_target column 2 has near-zero norm")
    z_ins = z_ins / z_norm

    # +Z points towards the eye; moving away from the eye is -Z.
    T_stage[:3, 3] -= float(extra_clearance_mm) * z_ins
    return T_stage


def confirm_robot_step(title: str, details=None) -> bool:
    """
    Ask the user for explicit confirmation before a robot-related step.

    Returns:
        True  -> execute the step
        False -> skip this step

    Raises:
        KeyboardInterrupt if the user chooses q/quit.
    """
    print("")
    print("────────────────────────────────────────────")
    print(f"[CONFIRM] {title}")

    if details is not None:
        if isinstance(details, (list, tuple)):
            for line in details:
                print(f"  {line}")
        else:
            print(f"  {details}")

    ans = input("ENTER = ejecutar | s = saltar | q = abortar: ").strip().lower()

    if ans in ("q", "quit", "exit"):
        raise KeyboardInterrupt(f"User aborted before: {title}")

    if ans in ("s", "skip", "saltar"):
        print(f"  [SKIP] {title}")
        print("────────────────────────────────────────────")
        return False

    print(f"  [OK] {title}")
    print("────────────────────────────────────────────")
    return True

# ─────────────────────────────────────────────
# CAPTURE HELPER
# ─────────────────────────────────────────────

def capture_frame(camera, settings):
    """
    Capture one Zivid frame and return:
        rgb: H x W x 3 uint8
        xyz: H x W x 3 float32
    """
    with camera.capture_2d_3d(settings) as frame:
        pc = frame.point_cloud()

        # Color as NumPy array: H x W x 4
        try:
            rgba = pc.copy_data("rgba_srgb")
        except Exception:
            rgba = pc.copy_data("rgba")

        rgb = rgba[:, :, :3].astype(np.uint8)

        # XYZ as NumPy array: H x W x 3, in mm
        xyz = pc.copy_data("xyz").astype(np.float32)

    return rgb, xyz


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    rviz_visualizer = make_rviz_visualizer()

    model_A = YOLO(str(MODEL_A_PATH))
    model_B = load_model_b(MODEL_B_PATH)

    print(f"Device: {DEVICE}")

    app = zivid.Application()

    print("Connecting to camera...")
    camera = app.connect_camera()

    print(
        f"Camera: {camera.info.model_name} | "
        f"Serial: {camera.info.serial_number}\n"
    )

    settings = zivid.Settings.load(str(SETTINGS_YML))

    base_name = input("Capture name (or ENTER for 'test'): ").strip() or "test"
    names = [f"{base_name}_{i:03d}" for i in range(1, 999)]

    input("\nPress ENTER to start...\n")

    name_iter = iter(names)

    for nombre in name_iter:
        retries = 0
        trocar_ok = False

        capture_dir = OUT_DIR / nombre
        capture_dir.mkdir(parents=True, exist_ok=True)

        last_rgb = None
        last_xyz = None
        last_result = None

        while not trocar_ok:
            retry_msg = f" [retry {retries}/{MAX_RETRIES}]" if retries else ""
            print(f"Capturing {nombre}...{retry_msg}")

            rgb, xyz = capture_frame(
                camera=camera,
                settings=settings,
            )

            last_rgb = rgb
            last_xyz = xyz

            t0 = time.perf_counter()
            planning_time_ms = None

            result = run_pipeline(
                rgb=rgb,
                xyz=xyz,
                model_A=model_A,
                model_B=model_B,
            )

            if result is not None and result["pose"] is not None:
                pose = result["pose"]

                # Save prediction.json immediately, before robot conversion.
                # The second-stage script reads p_pre_mm + z_axis_cam/v_trocar from this file.
                save_evaluation_prediction(result, capture_dir)
                PREDICTION_PATH = capture_dir / "prediction.json"

                T_BASE_TARGET_PATH = capture_dir / "T_base_target.npy"
                meca_pose_script = get_meca_pose_script_path()

                cmd_robot_pose = [
                    "python3",
                    str(meca_pose_script),
                    "--prediction",
                    str(PREDICTION_PATH),
                    "--out-dir",
                    str(capture_dir),
                ]

                print("  building robot target with q_align * default_orientation...")
                print("  " + " ".join(cmd_robot_pose))

                try:
                    subprocess.run(
                        cmd_robot_pose,
                        check=True,
                    )

                    if not T_BASE_TARGET_PATH.exists():
                        raise FileNotFoundError(
                            f"Robot pose script finished but did not create: {T_BASE_TARGET_PATH}"
                        )

                    T_base_target = np.load(str(T_BASE_TARGET_PATH)).astype(np.float64)
                    pose["T_base_target"] = T_base_target

                    print("  target in robot/base frame (T_target, native MovePose, 50 mm clearance):")
                    print(
                        f"  X={T_base_target[0, 3]:.2f} "
                        f"Y={T_base_target[1, 3]:.2f} "
                        f"Z={T_base_target[2, 3]:.2f} mm"
                    )

                    T_base_stage = make_stage_matrix(
                        T_base_target,
                        MOVEIT_PRE_CLEARANCE_MM,
                    )
                    T_BASE_STAGE_PATH = capture_dir / "T_base_stage.npy"
                    np.save(str(T_BASE_STAGE_PATH), T_base_stage.astype(np.float64))
                    print(
                        f"  MoveIt staging target (T_stage): "
                        f"extra_clearance={MOVEIT_PRE_CLEARANCE_MM:.1f} mm -> "
                        f"X={T_base_stage[0, 3]:.2f} "
                        f"Y={T_base_stage[1, 3]:.2f} "
                        f"Z={T_base_stage[2, 3]:.2f} mm"
                    )

                    if rviz_visualizer is not None:
                        print("  publishing T_base_target markers in RViz...")
                        rviz_visualizer.publish_target_pose(T_base_target)

                        T_BASE_CAM_PATH = capture_dir / "T_base_cam.npy"
                        if T_BASE_CAM_PATH.exists():
                            try:
                                T_base_cam = np.load(str(T_BASE_CAM_PATH)).astype(np.float64)
                                print("  publishing eye_debug markers in RViz using T_base_cam.npy...")
                                rviz_visualizer.publish_eye_and_axis(pose, T_base_cam)
                            except Exception as e:
                                print(f"  [RViz] eye_debug NOT published: {e}")
                        else:
                            print(
                                "  [RViz] eye_debug NOT published: "
                                f"{T_BASE_CAM_PATH} not found. "
                                "Use the updated build_meca_pose_from_axis_z.py that saves T_base_cam.npy."
                            )

                    if AUTO_PLAN:
                        cmd = [
                            "python3",
                            str(MOVEIT_PLAN_SCRIPT),
                            "--matrix",
                            str(T_BASE_STAGE_PATH),
                            "--units",
                            "mm",
                            "--group",
                            "meca_arm",
                            "--base-frame",
                            "meca_base_link",
                            "--target-link",
                            "tcp_link",
                            "--velocity-scaling",
                            str(MOVEIT_VELOCITY_SCALING),
                            "--acceleration-scaling",
                            str(MOVEIT_ACCELERATION_SCALING),
                        ]

                        if MOVEIT_PLAN_ONLY:
                            cmd.append("--plan-only")

                        if MOVEIT_DISABLE_AXIS_CORRECTION:
                            cmd.append("--disable-axis-correction")

                        mode = "PLAN ONLY" if MOVEIT_PLAN_ONLY else "PLAN + EXECUTE"
                        axis_mode = (
                            "axis correction DISABLED"
                            if MOVEIT_DISABLE_AXIS_CORRECTION
                            else "axis correction ENABLED"
                        )

                        print(
                            "  prepared MoveIt plan/execute script to T_stage "
                            "using T_base_stage.npy directly."
                        )
                        print(f"  mode: {mode}")
                        print(f"  orientation: {axis_mode}")
                        print("  " + " ".join(cmd))

                        plan_t0 = time.perf_counter()
                        try:
                            moveit_ok = False
                            native_movepose_ok = False

                            moveit_allowed = True
                            if REQUIRE_ENTER_BEFORE_MOVEIT:
                                moveit_allowed = confirm_robot_step(
                                    "MoveIt -> T_stage",
                                    details=[
                                        f"T_stage = T_target + {MOVEIT_PRE_CLEARANCE_MM:.1f} mm hacia fuera del ojo",
                                        f"mode: {mode}",
                                        f"orientation: {axis_mode}",
                                        "command:",
                                        "  " + " ".join(cmd),
                                    ],
                                )

                            if moveit_allowed:
                                subprocess.run(
                                    cmd,
                                    check=True,
                                )
                                moveit_ok = True
                                print("  MoveIt plan/execute script finished successfully")
                            else:
                                print("  MoveIt step skipped by user.")

                            if moveit_ok:
                                if AUTO_NATIVE_MOVEPOSE and not MOVEIT_PLAN_ONLY:
                                    native_script = get_native_movepose_script_path()
                                    cmd_native = [
                                        "python3",
                                        str(native_script),
                                        "--target",
                                        str(T_BASE_TARGET_PATH),
                                        "--extra-clearance-mm",
                                        str(NATIVE_MOVEPOSE_EXTRA_CLEARANCE_MM),
                                    ]

                                    if NATIVE_MOVEPOSE_DRY_RUN:
                                        cmd_native.append("--dry-run")

                                    print(
                                        "  prepared native Meca500 MovePose to T_target "
                                        "using robot FK/IK."
                                    )
                                    print("  " + " ".join(cmd_native))

                                    native_allowed = True
                                    if REQUIRE_ENTER_BEFORE_NATIVE_MOVEPOSE:
                                        native_allowed = confirm_robot_step(
                                            "Meca MovePose nativo -> T_target",
                                            details=[
                                                "T_target = T_base_target.npy, clearance final 50 mm",
                                                "usa FK/IK nativa del robot, no FK del URDF",
                                                "command:",
                                                "  " + " ".join(cmd_native),
                                            ],
                                        )

                                    if native_allowed:
                                        subprocess.run(
                                            cmd_native,
                                            check=True,
                                        )
                                        native_movepose_ok = not NATIVE_MOVEPOSE_DRY_RUN

                                        if NATIVE_MOVEPOSE_DRY_RUN:
                                            print("  Native MovePose dry-run finished; robot was not moved.")
                                        else:
                                            print("  Native MovePose command published successfully")
                                    else:
                                        print("  Native MovePose skipped by user.")

                                elif MOVEIT_PLAN_ONLY:
                                    print("  Native MovePose skipped because MOVEIT_PLAN_ONLY=True")

                                else:
                                    print("  Native MovePose skipped because AUTO_NATIVE_MOVEPOSE=False")

                                advance_ok = False
                                if AUTO_ADVANCE_MOVELIN and not MOVEIT_PLAN_ONLY:
                                    if not native_movepose_ok:
                                        print(
                                            "  MoveLin advance skipped: native MovePose was not executed."
                                        )
                                    else:
                                        movelin_script = get_movelin_advance_script_path()
                                        cmd_advance = [
                                            "python3",
                                            str(movelin_script),
                                            "--distance-mm",
                                            str(ADVANCE_MM),
                                            "--axis-column",
                                            str(ADVANCE_AXIS_COLUMN),
                                            "--axis-sign",
                                            str(ADVANCE_AXIS_SIGN),
                                        ]

                                        print(
                                            "  prepared final MoveLin advance from CURRENT TCP pose."
                                        )
                                        print(
                                            f"  advance: {ADVANCE_MM:.1f} mm, "
                                            f"axis_column={ADVANCE_AXIS_COLUMN}, "
                                            f"axis_sign={ADVANCE_AXIS_SIGN:+.0f}"
                                        )
                                        print("  " + " ".join(cmd_advance))

                                        advance_allowed = True
                                        if REQUIRE_ENTER_BEFORE_MOVELIN_ADVANCE:
                                            advance_allowed = confirm_robot_step(
                                                "Meca MoveLin advance desde TCP actual",
                                                details=[
                                                    f"advance={ADVANCE_MM:.1f} mm",
                                                    f"axis_column={ADVANCE_AXIS_COLUMN}",
                                                    f"axis_sign={ADVANCE_AXIS_SIGN:+.0f}",
                                                    "command:",
                                                    "  " + " ".join(cmd_advance),
                                                ],
                                            )

                                        if advance_allowed:
                                            subprocess.run(
                                                cmd_advance,
                                                check=True,
                                            )
                                            advance_ok = True
                                            print("  MoveLin advance command published successfully")
                                        else:
                                            print("  MoveLin advance skipped by user.")

                                # ── Rotación interactiva J6 (alineación de aguja) ──
                                j6_confirmed = True  # True si se omite el paso J6
                                if advance_ok and AUTO_J6_ROTATION_STEP and not MOVEIT_PLAN_ONLY:
                                    j6_script = get_j6_rotate_script_path()
                                    print()
                                    print(
                                        "  Iniciando sesión interactiva de rotación J6 "
                                        "para alineación de aguja..."
                                    )
                                    j6_proc = subprocess.run(
                                        ["python3", str(j6_script)],
                                        check=False,
                                    )
                                    j6_confirmed = (j6_proc.returncode == 0)
                                    if j6_confirmed:
                                        print("  Sesión de rotación J6 confirmada.")
                                    else:
                                        print(
                                            "  Sesión de rotación J6 cancelada (q). "
                                            "MoveLin final omitido."
                                        )

                                # ── MoveLin de inserción final ──────────────────────
                                if (
                                    advance_ok
                                    and j6_confirmed
                                    and AUTO_FINAL_MOVELIN_5MM
                                    and not MOVEIT_PLAN_ONLY
                                ):
                                    movelin_script_final = get_movelin_advance_script_path()
                                    cmd_final = [
                                        "python3",
                                        str(movelin_script_final),
                                        "--distance-mm",
                                        str(FINAL_MOVELIN_MM),
                                        "--axis-column",
                                        str(ADVANCE_AXIS_COLUMN),
                                        "--axis-sign",
                                        str(ADVANCE_AXIS_SIGN),
                                    ]

                                    print(
                                        f"  prepared final {FINAL_MOVELIN_MM:.1f} mm insertion MoveLin."
                                    )
                                    print("  " + " ".join(cmd_final))

                                    final_allowed = True
                                    if REQUIRE_ENTER_BEFORE_FINAL_MOVELIN:
                                        final_allowed = confirm_robot_step(
                                            f"MoveLin inserción final {FINAL_MOVELIN_MM:.1f} mm",
                                            details=[
                                                f"advance={FINAL_MOVELIN_MM:.1f} mm",
                                                f"axis_column={ADVANCE_AXIS_COLUMN}",
                                                f"axis_sign={ADVANCE_AXIS_SIGN:+.0f}",
                                                "command:",
                                                "  " + " ".join(cmd_final),
                                            ],
                                        )

                                    if final_allowed:
                                        subprocess.run(
                                            cmd_final,
                                            check=True,
                                        )
                                        print(
                                            f"  MoveLin final {FINAL_MOVELIN_MM:.1f} mm "
                                            "publicado correctamente."
                                        )
                                    else:
                                        print("  MoveLin final saltado por el usuario.")
                            else:
                                print(
                                    "  Native MovePose and MoveLin advance skipped because "
                                    "MoveIt stage was not executed successfully."
                                )

                        except subprocess.CalledProcessError as e:
                            print(f"  Robot execution step failed with code {e.returncode}")

                        finally:
                            planning_time_ms = elapsed_ms(plan_t0)

                except subprocess.CalledProcessError as e:
                    print(f"  robot-pose script failed with code {e.returncode}")
                    print("  MoveIt planning skipped")

                except Exception as e:
                    print(f"  robot target not computed: {e}")
                    print("  MoveIt planning skipped")


            dt = elapsed_ms(t0)

            if result is not None:
                timings_ms = result.setdefault("timings_ms", {})
                timings_ms["planning_time_ms"] = planning_time_ms
                timings_ms["total_pipeline_time_ms"] = dt

            last_result = result

            # ── Print detection results ───────────────────────
            if result is not None:
                print(f"  eye:    {result['eye_bbox']}  conf={result['conf_A']}")

                if result["trocar_global"] is not None:
                    print(f"  trocar: {result['trocar_global']}")

                    if result["trocar_3d_mm"] is not None:
                        x, y, z = result["trocar_3d_mm"]
                        print(f"  3D:     X={x:.2f}  Y={y:.2f}  Z={z:.2f} mm")
                    else:
                        print("  3D:     unavailable (invalid points)")

                else:
                    print("  trocar: not detected")

                if result["pose"] is not None:
                    pose = result["pose"]

                    p_eye = pose["p_entry_eye_mm"]
                    p_trocar = pose["p_entry_trocar_mm"]
                    p_pre = get_pose_p_pre(pose)
                    v = pose["v_trocar"]
                    c = pose["eye_center_mm"]
                    r = pose["eye_radius_mm"]

                    print(
                        f"  p_entry_eye:    "
                        f"X={p_eye[0]:.2f} Y={p_eye[1]:.2f} Z={p_eye[2]:.2f} mm"
                    )

                    print(
                        f"  p_entry_trocar: "
                        f"X={p_trocar[0]:.2f} Y={p_trocar[1]:.2f} Z={p_trocar[2]:.2f} mm"
                    )

                    print(
                        f"  p_pre:          "
                        f"X={p_pre[0]:.2f} Y={p_pre[1]:.2f} Z={p_pre[2]:.2f} mm"
                    )

                    print(
                        f"  axis:   "
                        f"vx={v[0]:.3f} vy={v[1]:.3f} vz={v[2]:.3f}"
                    )

                    print(
                        f"  eye:    "
                        f"C=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) "
                        f"r={r:.2f} mm"
                    )

                else:
                    print("  pose:   not estimated")

            else:
                print("  No detection (eye not found).")

            timings_for_print = result.get("timings_ms") if result is not None else {
                "planning_time_ms": planning_time_ms,
                "total_pipeline_time_ms": dt,
            }
            print_pipeline_timings(timings_for_print)
            print()

            # ── Retry logic ───────────────────────────────────
            trocar_ok = (
                result is not None
                and result["trocar_global"] is not None
                and result["pose"] is not None
            )

            if not trocar_ok:
                if retries < MAX_RETRIES:
                    retries += 1

                    if result is None:
                        reason = "eye not detected"
                    elif result["trocar_global"] is None:
                        reason = "trocar not detected"
                    else:
                        reason = "valid detection not obtained"

                    print(
                        f"  ⚠ {reason} — retrying automatically "
                        f"({retries}/{MAX_RETRIES})...\n"
                    )
                    continue

                print(
                    f"  ✗ Valid detection not obtained after {MAX_RETRIES} retries — "
                    f"moving on.\n"
                )
                break

        # ── Save final visualization and 3D debug outputs ─────
        if last_rgb is not None:
            vis = draw_results(
                cv2.cvtColor(last_rgb, cv2.COLOR_RGB2BGR),
                last_result,
            )

            cv2.imwrite(
                str(capture_dir / f"{nombre}_result.png"),
                vis,
            )

            save_debug_outputs(
                nombre=nombre,
                rgb=last_rgb,
                xyz=last_xyz,
                result=last_result,
                out_dir=capture_dir,
            )

            print(f"  debug outputs saved in: {capture_dir}")

        # ── Ask user to continue ──────────────────────────────
        continuar = input("ENTER = next | q = quit: ").strip().lower()

        if continuar == "q":
            break

    if rviz_visualizer is not None and ROS2_MARKERS_AVAILABLE:
        rviz_visualizer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print("\nDone.")


if __name__ == "__main__":
    main()
