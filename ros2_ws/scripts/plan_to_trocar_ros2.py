#!/usr/bin/env python3
"""
plan_to_trocar_ros2_moveit_execute.py

Plan and optionally execute a MoveIt 2 target pose for the Meca500 TCP.

Architecture for the real robot setup:

    T_base_target.npy
        ↓
    this script sends a MoveGroup goal to /move_action
        ↓
    MoveIt plans from the current /joint_states
        ↓
    if --plan-only is NOT used, MoveIt executes the trajectory
        ↓
    MoveIt calls /meca_arm_controller/follow_joint_trajectory
        ↓
    convert_to_meca_node executes MoveJoints waypoint by waypoint

Important:
    - This script does NOT publish to /joint_targets.
    - Execution is delegated to MoveIt.
    - MoveIt will use the controller configured in the MoveIt controller YAML.
    - Positions from vision are normally in millimetres.
    - ROS/MoveIt expects metres.
    - Quaternions are [x, y, z, w].

Inputs, in priority order:
    1) --matrix <path/to/T_base_target.npy>
    2) --data-dir containing T_base_target.npy
    3) --data-dir containing p_pre_base_mm.npy + approach_orientation_xyzw.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    BoundingVolume,
    MoveItErrorCodes,
)
from shape_msgs.msg import SolidPrimitive


# ─────────────────────────────────────────────────────────────────────────────
# Frame correction: Meca500 target frame → MoveIt tcp_link convention
# ─────────────────────────────────────────────────────────────────────────────
# Keep this correction enabled if it is the one you validated in your previous
# planning tests. Disable it with --disable-axis-correction only for debugging.
_MECA500_FRF_AXIS_CORRECTION = np.array([
    [0.0, 0.0, -1.0],
    [0.0, 1.0,  0.0],
    [1.0, 0.0,  0.0],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Quaternion / pose utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_quaternion_xyzw(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Quaternion has near-zero norm")
    return q / n


def quaternion_xyzw_to_rotation_matrix(q):
    """Convert quaternion [x, y, z, w] to a 3x3 rotation matrix."""
    x, y, z, w = normalize_quaternion_xyzw(q)

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def rotation_matrix_to_quaternion_xyzw(R):
    """
    Convert a 3x3 rotation matrix to quaternion [x, y, z, w].
    No scipy dependency.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"Rotation matrix must be 3x3, got {R.shape}")

    tr = float(np.trace(R))

    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return normalize_quaternion_xyzw(np.array([x, y, z, w], dtype=np.float64))


def validate_rotation_matrix(R, name="R"):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"{name} must be 3x3, got {R.shape}")

    det = float(np.linalg.det(R))
    if not np.isfinite(det):
        raise ValueError(f"{name} determinant is not finite")

    if abs(det - 1.0) > 1e-2:
        raise ValueError(f"{name} is not a valid rotation matrix. det={det:.6f}")


def make_pose_stamped(base_frame, position_m, quat_xyzw, clock):
    position_m = np.asarray(position_m, dtype=np.float64).reshape(3)
    quat_xyzw = normalize_quaternion_xyzw(quat_xyzw)

    ps = PoseStamped()
    ps.header.frame_id = base_frame
    ps.header.stamp = clock.now().to_msg()

    ps.pose.position.x = float(position_m[0])
    ps.pose.position.y = float(position_m[1])
    ps.pose.position.z = float(position_m[2])

    ps.pose.orientation.x = float(quat_xyzw[0])
    ps.pose.orientation.y = float(quat_xyzw[1])
    ps.pose.orientation.z = float(quat_xyzw[2])
    ps.pose.orientation.w = float(quat_xyzw[3])

    return ps


# ─────────────────────────────────────────────────────────────────────────────
# Target loading
# ─────────────────────────────────────────────────────────────────────────────

def maybe_apply_axis_correction(R, args):
    validate_rotation_matrix(R, "R_target")

    if args.disable_axis_correction:
        return R

    R_corrected = R @ _MECA500_FRF_AXIS_CORRECTION
    validate_rotation_matrix(R_corrected, "R_target_corrected")
    return R_corrected




def apply_pre_clearance_to_position(position_native_units, R_native, args, *, units_are_mm: bool):
    """
    Move the target backwards along the insertion axis before planning.

    Convention:
        R_native[:, 2] is +Z insertion axis, pointing towards the eye.
        Pre-clearance moves away from the eye, i.e. along -Z.

    Args:
        position_native_units: position in the same units as the loaded target.
        R_native: target rotation before MoveIt axis correction.
        units_are_mm: True if position is in mm, False if position is in m.
    """
    position = np.asarray(position_native_units, dtype=np.float64).reshape(3).copy()
    pre_clearance_mm = float(getattr(args, "pre_clearance_mm", 0.0))

    if abs(pre_clearance_mm) < 1e-12:
        return position

    z_ins = np.asarray(R_native, dtype=np.float64)[:, 2]
    z_norm = np.linalg.norm(z_ins)
    if z_norm < 1e-9:
        raise ValueError("Target insertion axis R[:, 2] has near-zero norm")
    z_ins = z_ins / z_norm

    delta = pre_clearance_mm if units_are_mm else pre_clearance_mm * 0.001
    return position - delta * z_ins

def load_target_from_files(args):
    """
    Returns:
        position_m, quat_xyzw, source_description
    """
    units = args.units.lower()
    if units not in ("mm", "m"):
        raise ValueError("--units must be either 'mm' or 'm'")

    scale = 0.001 if units == "mm" else 1.0

    # 1) Explicit matrix path.
    if args.matrix is not None:
        matrix_path = Path(args.matrix)
        T = np.asarray(np.load(matrix_path), dtype=np.float64)

        if T.shape != (4, 4):
            raise ValueError(f"Matrix must be 4x4, got {T.shape}: {matrix_path}")

        R_native = T[:3, :3]
        position_native = apply_pre_clearance_to_position(
            T[:3, 3],
            R_native,
            args,
            units_are_mm=(units == "mm"),
        )
        position_m = position_native * scale
        R = maybe_apply_axis_correction(R_native, args)
        quat_xyzw = rotation_matrix_to_quaternion_xyzw(R)
        return position_m, quat_xyzw, f"matrix: {matrix_path}"

    data_dir = Path(args.data_dir)

    # 2) T_base_target.npy in data_dir.
    T_path = data_dir / "T_base_target.npy"
    if T_path.exists():
        T = np.asarray(np.load(T_path), dtype=np.float64)

        if T.shape != (4, 4):
            raise ValueError(f"T_base_target.npy must be 4x4, got {T.shape}: {T_path}")

        R_native = T[:3, :3]
        position_native = apply_pre_clearance_to_position(
            T[:3, 3],
            R_native,
            args,
            units_are_mm=(units == "mm"),
        )
        position_m = position_native * scale
        R = maybe_apply_axis_correction(R_native, args)
        quat_xyzw = rotation_matrix_to_quaternion_xyzw(R)
        return position_m, quat_xyzw, f"data-dir matrix: {T_path}"

    # 3) p_pre_base_mm.npy + approach_orientation_xyzw.npy in data_dir.
    p_path = data_dir / "p_pre_base_mm.npy"
    q_path = data_dir / "approach_orientation_xyzw.npy"

    if not p_path.exists() or not q_path.exists():
        raise FileNotFoundError(
            "Could not find target files. Expected one of:\n"
            f"  --matrix <path/to/T_base_target.npy>\n"
            f"  {T_path}\n"
            f"  {p_path} + {q_path}"
        )

    # This file name explicitly says mm, so always convert from mm.
    position_mm = np.asarray(np.load(p_path), dtype=np.float64).reshape(3)
    q_raw = normalize_quaternion_xyzw(np.load(q_path))
    R_native = quaternion_xyzw_to_rotation_matrix(q_raw)
    position_mm = apply_pre_clearance_to_position(
        position_mm,
        R_native,
        args,
        units_are_mm=True,
    )
    position_m = position_mm * 0.001
    R = maybe_apply_axis_correction(R_native, args)
    quat_xyzw = rotation_matrix_to_quaternion_xyzw(R)

    return position_m, quat_xyzw, f"data-dir arrays: {p_path.name} + {q_path.name}"


# ─────────────────────────────────────────────────────────────────────────────
# MoveIt action planner
# ─────────────────────────────────────────────────────────────────────────────

class MecaTrocarPlanner(Node):
    def __init__(self, args):
        super().__init__("meca_trocar_planner")
        self.args = args
        # Flag de resultado: True solo si MoveIt planificó Y ejecutó con éxito.
        # main() lo lee para decidir el exit code del proceso.
        self._success = False
        self._action_client = ActionClient(self, MoveGroup, args.action_name)

        position_m, quat_xyzw, source = load_target_from_files(args)

        self.target_pose = make_pose_stamped(
            base_frame=args.base_frame,
            position_m=position_m,
            quat_xyzw=quat_xyzw,
            clock=self.get_clock(),
        )

        self.get_logger().info(f"Loaded target from: {source}")
        self.get_logger().info(
            "Planning target pose:\n"
            f"  frame       = {args.base_frame}\n"
            f"  target_link = {args.target_link}\n"
            f"  group       = {args.group}\n"
            f"  action      = {args.action_name}\n"
            f"  position    = "
            f"[{position_m[0]:.4f}, {position_m[1]:.4f}, {position_m[2]:.4f}] m\n"
            f"  pre_clearance = {float(args.pre_clearance_mm):.3f} mm\n"
            f"  quaternion  = "
            f"[{quat_xyzw[0]:.6f}, {quat_xyzw[1]:.6f}, "
            f"{quat_xyzw[2]:.6f}, {quat_xyzw[3]:.6f}] xyzw"
        )

    def build_goal_constraints(self):
        c = Constraints()
        c.name = "tcp_target_pose"

        # Position constraint for target_link.
        pc = PositionConstraint()
        pc.header.frame_id = self.args.base_frame
        pc.header.stamp = self.get_clock().now().to_msg()
        pc.link_name = self.args.target_link
        pc.weight = 1.0

        bv = BoundingVolume()
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX

        # SolidPrimitive.BOX dimensions are full box lengths.
        # If position_tolerance_m = 0.002, accepted region is roughly +/-2 mm.
        tol = float(self.args.position_tolerance_m)
        box.dimensions = [2.0 * tol, 2.0 * tol, 2.0 * tol]

        center_pose = Pose()
        center_pose.position = self.target_pose.pose.position
        center_pose.orientation.w = 1.0

        bv.primitives.append(box)
        bv.primitive_poses.append(center_pose)
        pc.constraint_region = bv
        c.position_constraints.append(pc)

        # Orientation constraint for target_link.
        oc = OrientationConstraint()
        oc.header.frame_id = self.args.base_frame
        oc.header.stamp = self.get_clock().now().to_msg()
        oc.link_name = self.args.target_link
        oc.orientation = self.target_pose.pose.orientation
        oc.absolute_x_axis_tolerance = float(self.args.orientation_tolerance_rad)
        oc.absolute_y_axis_tolerance = float(self.args.orientation_tolerance_rad)
        oc.absolute_z_axis_tolerance = float(self.args.orientation_tolerance_rad)
        oc.weight = 1.0
        c.orientation_constraints.append(oc)

        return c

    def send_goal(self):
        self.get_logger().info(
            f"Waiting for MoveIt action server: {self.args.action_name}"
        )

        if not self._action_client.wait_for_server(
            timeout_sec=float(self.args.server_timeout_sec)
        ):
            self.get_logger().error("MoveIt action server not available.")
            rclpy.shutdown()
            sys.exit(1)

        req = MotionPlanRequest()
        req.group_name = self.args.group
        req.num_planning_attempts = int(self.args.planning_attempts)
        req.allowed_planning_time = float(self.args.allowed_planning_time)
        req.pipeline_id = self.args.pipeline_id
        req.planner_id = self.args.planner_id
        req.max_velocity_scaling_factor = float(self.args.velocity_scaling)
        req.max_acceleration_scaling_factor = float(self.args.acceleration_scaling)

        # Use current /joint_states as the start state.
        # In the real architecture, /joint_states must come from convert_to_meca_node.
        req.start_state.is_diff = True

        req.goal_constraints.append(self.build_goal_constraints())

        goal_msg = MoveGroup.Goal()
        goal_msg.request = req

        # This is the key switch:
        #   plan_only=True   -> MoveIt only plans
        #   plan_only=False  -> MoveIt plans and executes using the configured controller
        goal_msg.planning_options.plan_only = bool(self.args.plan_only)
        goal_msg.planning_options.look_around = False
        goal_msg.planning_options.replan = bool(self.args.replan)
        goal_msg.planning_options.replan_attempts = int(self.args.replan_attempts)

        mode = "PLAN ONLY" if self.args.plan_only else "PLAN + EXECUTE"
        self.get_logger().info(f"Sending MoveIt goal ({mode})...")

        future = self._action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error("No goal handle returned by MoveIt.")
            rclpy.shutdown()
            sys.exit(1)

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected by MoveIt.")
            rclpy.shutdown()
            sys.exit(1)

        self.get_logger().info("Goal accepted by MoveIt. Waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        action_result = future.result()

        if action_result is None:
            self.get_logger().error("No result returned by MoveIt.")
            rclpy.shutdown()
            sys.exit(1)

        result = action_result.result
        code = result.error_code.val

        if code == MoveItErrorCodes.SUCCESS:
            if self.args.plan_only:
                self.get_logger().info(
                    "Planning succeeded. No execution requested (--plan-only)."
                )
            else:
                self.get_logger().info(
                    "Motion completed successfully. MoveIt should have executed via "
                    "/meca_arm_controller/follow_joint_trajectory."
                )
            # Único camino que marca el script como exitoso.
            self._success = True
        else:
            self.get_logger().error(f"MoveIt failed. error_code.val = {code}")
            self.get_logger().error(
                "Useful checks:\n"
                "  1) Is /joint_states published only by convert_to_meca_node?\n"
                "  2) Does ros2 action list show /meca_arm_controller/follow_joint_trajectory?\n"
                "  3) Is target_link really tcp_link in the URDF/SRDF?\n"
                "  4) Try --target-link <flange_link> to separate TCP-offset issues.\n"
                "  5) Try larger --position-tolerance-m 0.005 and "
                "--orientation-tolerance-rad 0.1."
            )
            rclpy.shutdown()
            sys.exit(1)

        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plan/execute a MoveIt pose target for the Meca500 TCP."
    )

    parser.add_argument(
        "--data-dir",
        default=".",
        help="Directory containing T_base_target.npy, or p_pre_base_mm.npy + approach_orientation_xyzw.npy.",
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Optional explicit path to T_base_target.npy. Overrides --data-dir.",
    )
    parser.add_argument(
        "--units",
        default="mm",
        choices=["mm", "m"],
        help="Units of the matrix translation. Vision output normally uses mm.",
    )
    parser.add_argument(
        "--pre-clearance-mm",
        type=float,
        default=0.0,
        help=(
            "Move the target backwards by N mm along -Z of T_base_target before "
            "planning with MoveIt. Use 30 to plan to T_stage when T_base_target "
            "is the 50 mm native target. Applied before axis correction."
        ),
    )

    parser.add_argument("--base-frame", default="meca_base_link")
    parser.add_argument("--target-link", default="tcp_link")
    parser.add_argument("--group", default="meca_arm")
    parser.add_argument("--action-name", default="/move_action")

    parser.add_argument("--pipeline-id", default="ompl")
    parser.add_argument("--planner-id", default="RRTConnectkConfigDefault")
    parser.add_argument("--planning-attempts", type=int, default=10)
    parser.add_argument("--allowed-planning-time", type=float, default=10.0)
    parser.add_argument("--velocity-scaling", type=float, default=0.10)
    parser.add_argument("--acceleration-scaling", type=float, default=0.10)

    parser.add_argument(
        "--position-tolerance-m",
        type=float,
        default=0.002,
        help="Position tolerance in metres. 0.002 means roughly +/-2 mm.",
    )
    parser.add_argument(
        "--orientation-tolerance-rad",
        type=float,
        default=0.03,
        help="Orientation tolerance in radians. 0.03 rad is about 1.7 deg.",
    )

    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Only plan, do not execute. Useful for the first RViz check.",
    )
    parser.add_argument(
        "--replan",
        action="store_true",
        help="Allow MoveIt replanning during execution.",
    )
    parser.add_argument("--replan-attempts", type=int, default=2)
    parser.add_argument("--server-timeout-sec", type=float, default=20.0)

    parser.add_argument(
        "--disable-axis-correction",
        action="store_true",
        help="Do not apply the Meca500→MoveIt orientation correction matrix.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    rclpy.init(args=None)
    node = None

    try:
        node = MecaTrocarPlanner(args)
        node.send_goal()
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Ctrl+C durante la planificación/ejecución: fallo desde el punto
        # de vista del pipeline (el robot no llegó a T_stage).
        print("[plan_to_trocar] Interrumpido por el usuario (Ctrl+C).", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(1)
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

    # rclpy.spin() terminó porque un callback llamó rclpy.shutdown().
    # Si fue por sys.exit(1) dentro de un callback, no llegamos aquí.
    # Si llegamos aquí, comprobamos el flag de éxito.
    if node is None or not node._success:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
