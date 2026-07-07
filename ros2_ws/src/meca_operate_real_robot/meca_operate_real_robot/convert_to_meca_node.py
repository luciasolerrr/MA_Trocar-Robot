#!/usr/bin/env python3

import threading
import queue
import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import JointState

from control_msgs.action import FollowJointTrajectory

from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster

from mecademicpy.robot import Robot

try:
    from scipy.spatial.transform import Rotation as SciPyRotation
except Exception:
    SciPyRotation = None


MECA_JOINT_NAMES = [
    "meca_axis_1_joint",
    "meca_axis_2_joint",
    "meca_axis_3_joint",
    "meca_axis_4_joint",
    "meca_axis_5_joint",
    "meca_axis_6_joint",
]


class JointCommandNode(Node):
    def __init__(self):
        super().__init__('convert_to_meca_node')

        # ─────────────────────────────────────
        # Parameters
        # ─────────────────────────────────────

        self.declare_parameter('robot_ip', '192.168.0.100')

        # TF frames
        self.declare_parameter('tf_parent_frame', 'meca_base_link')
        self.declare_parameter('tf_child_frame', 'needle_tcp')
        self.declare_parameter('tf_rate_hz', 60.0)

        # TCP / TRF definition
        # If the needle is aligned with flange +Z:
        #     SetTrf(0, 0, L, 0, 0, 0)
        # IMPORTANT: L must be in millimetres because Meca500 uses mm.
        self.declare_parameter('use_needle_tcp', True)
        self.declare_parameter('tcp_z_mm', 108.751)
        self.declare_parameter('tcp_x_mm', -0.053)
        self.declare_parameter('tcp_y_mm', -0.457)

        # Safe speed limits
        self.declare_parameter('joint_vel_percent', 1.0)
        self.declare_parameter('joint_acc_percent', 5.0)

        # Densification parameter for MoveIt trajectory waypoints.
        # If max joint jump between two consecutive waypoints is larger than this,
        # intermediate waypoints are inserted.
        self.declare_parameter('max_joint_step_deg', 5.0)

        # Decimation parameter for MoveIt trajectory waypoints.
        # MoveIt may output hundreds of very close waypoints. Because this V1
        # executor calls MoveJoints + WaitIdle for each accepted waypoint,
        # executing every tiny waypoint can be much slower than MoveIt's
        # expected trajectory duration. A value of 2.0 means: only execute a
        # new waypoint if at least one joint differs by >= 2 deg from the
        # previously executed waypoint. The final waypoint is always kept.
        # Set to 0.0 to disable decimation.
        self.declare_parameter('min_waypoint_delta_deg', 2.0)

        # If True, WaitIdle() is kept inside robot_lock. This is safest for
        # mecademicpy thread-safety, but /joint_states may freeze during each
        # waypoint. If False, WaitIdle() is outside the lock so the monitor can
        # keep reading the robot during motion. Test carefully before using False
        # on the real robot.
        self.declare_parameter('wait_idle_holds_lock', True)

        # After all MoveIt waypoints are executed, GetJoints() is called and
        # the final position is compared to the last waypoint.
        # If any joint differs by more than this value (degrees), the trajectory
        # is reported as FAILED so MoveIt propagates the error back to the pipeline.
        # Set to 0.0 to disable the check.
        self.declare_parameter('final_joint_tolerance_deg', 2.0)

        self.declare_parameter('cart_lin_vel_mm_s', 5.0)
        self.declare_parameter('cart_acc_mm_s2', 20.0)

        # Cartesian motion mode:
        #   "pose" -> MovePose
        #   "lin"  -> MoveLin
        self.declare_parameter('cartesian_motion_mode', 'pose')

        robot_ip = self.get_parameter('robot_ip').value

        self.tf_parent_frame = self.get_parameter('tf_parent_frame').value
        self.tf_child_frame = self.get_parameter('tf_child_frame').value
        self.tf_rate_hz = float(self.get_parameter('tf_rate_hz').value)

        self.use_needle_tcp = bool(self.get_parameter('use_needle_tcp').value)
        self.tcp_z_mm = float(self.get_parameter('tcp_z_mm').value)
        self.tcp_x_mm = float(self.get_parameter('tcp_x_mm').value)
        self.tcp_y_mm = float(self.get_parameter('tcp_y_mm').value)

        self.joint_vel_percent = float(self.get_parameter('joint_vel_percent').value)
        self.joint_acc_percent = float(self.get_parameter('joint_acc_percent').value)
        self.max_joint_step_deg = float(self.get_parameter('max_joint_step_deg').value)
        self.min_waypoint_delta_deg = float(self.get_parameter('min_waypoint_delta_deg').value)
        self.wait_idle_holds_lock = bool(self.get_parameter('wait_idle_holds_lock').value)
        self.final_joint_tolerance_deg = float(
            self.get_parameter('final_joint_tolerance_deg').value
        )

        self.cart_lin_vel_mm_s = float(self.get_parameter('cart_lin_vel_mm_s').value)
        self.cart_acc_mm_s2 = float(self.get_parameter('cart_acc_mm_s2').value)

        self.cartesian_motion_mode = str(
            self.get_parameter('cartesian_motion_mode').value
        ).lower()

        if self.cartesian_motion_mode not in ("pose", "lin"):
            self.get_logger().warn(
                f"Invalid cartesian_motion_mode='{self.cartesian_motion_mode}'. "
                "Using 'pose'."
            )
            self.cartesian_motion_mode = "pose"

        self.tf_period = (
            1.0 / self.tf_rate_hz if self.tf_rate_hz > 0.0 else 0.0167
        )

        # ─────────────────────────────────────
        # Locks and caches
        # ─────────────────────────────────────

        self.robot_lock = threading.Lock()
        self.latest_pose_lock = threading.Lock()
        self.latest_joints_lock = threading.Lock()

        self.latest_pose = None
        self.latest_joints = None

        # Duplicate protection for relative /meca_lin_advance commands.
        # Relative commands are not idempotent: receiving the same message twice
        # would advance twice. If a command_id is provided as msg.data[3],
        # duplicates are ignored at callback time.
        self.last_meca_lin_advance_command_id = None
        self.last_meca_lin_advance_command_time = 0.0

        # Duplicate protection for absolute /meca_move_pose commands.
        # send_meca_move_pose_from_target publishes 3 copies for DDS reliability.
        # Without dedup, convert_to_meca_node queues all 3 and executes MovePose
        # three times to the same target (harmless but wastes ~15-20 s).
        # The publisher embeds command_id as msg.data[6] when present.
        self.last_meca_move_pose_command_id = None

        # Action callback group. This helps the node keep publishing state while
        # a trajectory action is being executed.
        self.action_callback_group = ReentrantCallbackGroup()

        # ─────────────────────────────────────
        # Subscribers
        # ─────────────────────────────────────

        # Legacy/manual joint command topic.
        # Expected units: degrees.
        self.sub_joints = self.create_subscription(
            Float64MultiArray,
            'joint_targets',
            self.cb_joints,
            10,
        )

        # Legacy/manual Cartesian command topic.
        # Expected position units: mm.
        self.sub_pose = self.create_subscription(
            PoseStamped,
            'pose_for_meca',
            self.cb_pose_stamped,
            10,
        )

        # Native Meca500 linear Cartesian command.
        # Expected data: [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg].
        # This bypasses quaternion/Euler ambiguity and always uses MoveLin.
        self.sub_meca_lin_pose = self.create_subscription(
            Float64MultiArray,
            'meca_lin_pose',
            self.cb_meca_lin_pose,
            10,
        )

        # Native Meca500 Cartesian MovePose command.
        # Expected data: [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg].
        # This uses the Meca500 controller native FK/IK and the active TRF.
        self.sub_meca_move_pose = self.create_subscription(
            Float64MultiArray,
            'meca_move_pose',
            self.cb_meca_move_pose,
            10,
        )


        # Relative native Meca500 linear advance from the CURRENT TCP pose.
        # Expected data: [distance_mm, axis_column, axis_sign]
        #   axis_column: 0=X, 1=Y, 2=Z of the current TCP frame
        #   axis_sign:   +1 or -1
        # This is preferred after MoveIt because it uses the robot's real
        # current TCP pose/orientation, avoiding matrix-vs-native mismatch.
        self.sub_meca_lin_advance = self.create_subscription(
            Float64MultiArray,
            'meca_lin_advance',
            self.cb_meca_lin_advance,
            10,
        )

        # ─────────────────────────────────────
        # Publishers
        # ─────────────────────────────────────

        self.joint_pub = self.create_publisher(
            JointState,
            'joint_states',
            10,
        )

        self.tf_broadcaster = TransformBroadcaster(self)

        # ─────────────────────────────────────
        # Robot connection
        # ─────────────────────────────────────

        # IMPORTANT:
        # Your installed mecademicpy version does not support
        # Robot(disconnect_on_exception=False). Using Robot() keeps compatibility.
        self.robot = Robot()

        self.get_logger().info(f'Connecting to Meca500 at {robot_ip}...')
        self.robot.Connect(address=robot_ip)

        # ── ActivateAndHome ───────────────────────────────────────────────────
        # FATAL: if activation or homing fails the robot is in an undefined
        # state and must not accept motion commands.  The node exits so the
        # launch system can surface the failure clearly (respawn will retry).
        try:
            self.get_logger().info('Activating and homing robot...')
            self.robot.ActivateAndHome()
            self.get_logger().info('Robot activated and homed.')
        except Exception as e:
            self.get_logger().fatal(
                f'ActivateAndHome FAILED: {e}. '
                'Node will exit to prevent motion with undefined robot state.'
            )
            raise SystemExit(1)

        # ── TCP / TRF configuration (FATAL on failure) ────────────────────────
        self.configure_robot()

        # ─────────────────────────────────────
        # FollowJointTrajectory action server
        # ─────────────────────────────────────
        #
        # This is the important part for MoveIt.
        # MoveIt will send planned trajectories to:
        #
        #   /meca_arm_controller/follow_joint_trajectory
        #
        # The trajectory contains joint positions in radians.
        # Meca500 MoveJoints expects degrees.
        #

        self.trajectory_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/meca_arm_controller/follow_joint_trajectory",
            execute_callback=self.execute_trajectory_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.action_callback_group,
        )

        self.get_logger().info(
            "FollowJointTrajectory action server ready on "
            "/meca_arm_controller/follow_joint_trajectory"
        )

        # ─────────────────────────────────────
        # Timer for TF + JointState publishing
        # ─────────────────────────────────────

        self.timer = self.create_timer(
            self.tf_period,
            self.publish_state,
        )

        # ─────────────────────────────────────
        # Motion worker queue for legacy topics
        # ─────────────────────────────────────

        self.q = queue.Queue()

        self.worker = threading.Thread(
            target=self.run,
            daemon=True,
        )
        self.worker.start()

        # ─────────────────────────────────────
        # Monitoring thread
        # ─────────────────────────────────────

        self.monitor_thread = threading.Thread(
            target=self.monitor_loop,
            daemon=True,
        )
        self.monitor_thread.start()

        # ── Startup complete ──────────────────────────────────────────────────
        # This line is only reached if ActivateAndHome AND configure_robot both
        # succeeded (TRF verified).  If you see this in the logs, the robot is
        # correctly configured and ready to receive commands.
        self.get_logger().info('convert_to_meca_node ready. SetTrf accepted.')

    # ─────────────────────────────────────────
    # Robot setup
    # ─────────────────────────────────────────

    def configure_robot(self):
        """
        Configure the Meca500 controller.

        CRITICAL path (raises SystemExit on failure):
            1. SetTrf  — sets the active TCP/TRF to the needle tip.
               If SetTrf raises an exception the controller rejected the
               command; the node exits immediately.
               Note: GetTrf() is not available in the installed mecademicpy
               version.  A successful SetTrf (no exception) is the
               authoritative confirmation that the TRF was accepted.

        Non-critical path (warns and continues):
            - SetJointVel / SetJointAcc
            - SetCartLinVel / SetCartAcc
            - SetMonitoringInterval
        """

        # ── CRITICAL: SetTrf ──────────────────────────────────────────────────
        if self.use_needle_tcp:
            self.get_logger().info(
                "Setting Meca500 TRF/TCP to needle tip: "
                f"SetTrf({self.tcp_x_mm:.3f}, {self.tcp_y_mm:.3f}, "
                f"{self.tcp_z_mm:.3f}, 0.0, 0.0, 0.0)"
            )

            try:
                self.robot.SetTrf(
                    self.tcp_x_mm,
                    self.tcp_y_mm,
                    self.tcp_z_mm,
                    0.0,
                    0.0,
                    0.0,
                )
            except Exception as e:
                self.get_logger().fatal(
                    f'SetTrf FAILED: {e}. '
                    'Cannot guarantee correct TCP frame. Node will exit.'
                )
                raise SystemExit(1)

            # GetTrf() is not available in the installed mecademicpy version.
            # SetTrf not raising an exception means the controller accepted the
            # command — that is the authoritative confirmation.
            self.get_logger().info(
                f'TRF set OK: '
                f'x={self.tcp_x_mm:.3f}, y={self.tcp_y_mm:.3f}, '
                f'z={self.tcp_z_mm:.3f} mm (no readback available)'
            )

        else:
            self.get_logger().warn(
                'use_needle_tcp is False. Cartesian poses will refer to '
                'the default flange TRF.'
            )

        # ── Non-critical: velocity / acceleration limits ───────────────────────
        try:
            self.robot.SetJointVel(self.joint_vel_percent)
        except Exception as e:
            self.get_logger().warn(f'SetJointVel failed (non-fatal): {e}')

        try:
            self.robot.SetJointAcc(self.joint_acc_percent)
        except Exception as e:
            self.get_logger().warn(f'SetJointAcc failed (non-fatal): {e}')

        try:
            self.robot.SetCartLinVel(self.cart_lin_vel_mm_s)
        except Exception as e:
            self.get_logger().warn(f'SetCartLinVel failed (non-fatal): {e}')

        try:
            self.robot.SetCartAcc(self.cart_acc_mm_s2)
        except Exception as e:
            self.get_logger().warn(f'SetCartAcc failed (non-fatal): {e}')

        try:
            self.robot.SetMonitoringInterval(self.tf_period)
        except Exception as e:
            self.get_logger().warn(f'SetMonitoringInterval failed (non-fatal): {e}')

        self.get_logger().info(
            f'Robot speed setup: '
            f'joint_vel={self.joint_vel_percent} %, '
            f'joint_acc={self.joint_acc_percent} %, '
            f'cart_lin_vel={self.cart_lin_vel_mm_s} mm/s, '
            f'cart_acc={self.cart_acc_mm_s2} mm/s^2, '
            f'max_joint_step={self.max_joint_step_deg} deg, '
            f'min_waypoint_delta={self.min_waypoint_delta_deg} deg, '
            f'wait_idle_holds_lock={self.wait_idle_holds_lock}'
        )

    # ─────────────────────────────────────────
    # Subscribers
    # ─────────────────────────────────────────

    def cb_joints(self, msg: Float64MultiArray):
        """
        Receive joint target.

        Expected:
            msg.data = [j1, j2, j3, j4, j5, j6]

        Units:
            degrees, because mecademicpy MoveJoints expects degrees.
        """
        if len(msg.data) != 6:
            self.get_logger().error(
                f'Need 6 joint values, received {len(msg.data)}.'
            )
            return

        joints_deg = tuple(float(x) for x in msg.data)

        self.get_logger().info(
            f'Received joint target deg: {joints_deg}'
        )

        self.q.put(("joint", joints_deg))

    def cb_meca_lin_pose(self, msg: Float64MultiArray):
        """
        Receive a native Meca500 Cartesian target for a linear motion.

        Expected:
            msg.data = [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg]

        Units/convention:
            - position in millimetres
            - Euler angles in degrees, Meca500 convention
            - executed with MoveLin regardless of cartesian_motion_mode
        """
        if len(msg.data) != 6:
            self.get_logger().error(
                f'Need 6 Cartesian values [x,y,z,a,b,c], received {len(msg.data)}.'
            )
            return

        goal = tuple(float(x) for x in msg.data)

        self.get_logger().info(
            "Received native MoveLin target: "
            f"x={goal[0]:.3f} mm, "
            f"y={goal[1]:.3f} mm, "
            f"z={goal[2]:.3f} mm, "
            f"a={goal[3]:.3f} deg, "
            f"b={goal[4]:.3f} deg, "
            f"c={goal[5]:.3f} deg"
        )

        self.q.put(("cart_lin", goal))

    def cb_meca_move_pose(self, msg: Float64MultiArray):
        """
        Receive a native Meca500 Cartesian target for MovePose.

        Expected:
            msg.data = [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg]
            msg.data = [x_mm, y_mm, z_mm, a_deg, b_deg, c_deg, command_id]  (optional)

        Units/convention:
            - position in millimetres
            - Euler angles in degrees, Meca500 intrinsic/mobile XYZ convention
            - executed with MovePose
            - target refers to the active TRF/TCP set by SetTrf

        The optional 7th element (command_id) enables duplicate suppression.
        send_meca_move_pose_from_target.py publishes 3 copies for DDS reliability;
        without dedup all 3 would be executed, wasting ~15-20 s.
        """
        if len(msg.data) not in (6, 7):
            self.get_logger().error(
                f'Need 6 or 7 Cartesian values [x,y,z,a,b,c,(command_id)], received {len(msg.data)}.'
            )
            return

        # Optional duplicate suppression via command_id (7th element).
        if len(msg.data) == 7:
            command_id = float(msg.data[6])
            if self.last_meca_move_pose_command_id == command_id:
                self.get_logger().warn(
                    f'Ignoring duplicate /meca_move_pose command_id={command_id:.0f}'
                )
                return
            self.last_meca_move_pose_command_id = command_id

        goal = tuple(float(x) for x in msg.data[:6])

        self.get_logger().info(
            "Received native MovePose target: "
            f"x={goal[0]:.3f} mm, "
            f"y={goal[1]:.3f} mm, "
            f"z={goal[2]:.3f} mm, "
            f"a={goal[3]:.3f} deg, "
            f"b={goal[4]:.3f} deg, "
            f"c={goal[5]:.3f} deg"
        )

        self.q.put(("cart_pose", goal))

    @staticmethod
    def _meca_intrinsic_xyz_to_matrix(a_deg, b_deg, c_deg):
        """
        Convert Meca500 Euler angles to a rotation matrix.

        Meca500 uses intrinsic/mobile XYZ Euler angles.
        scipy uppercase 'XYZ' matches intrinsic XYZ.
        """
        if SciPyRotation is not None:
            return SciPyRotation.from_euler(
                'XYZ',
                [float(a_deg), float(b_deg), float(c_deg)],
                degrees=True,
            ).as_matrix()

        # Fallback: intrinsic XYZ = Rx(a) @ Ry(b) @ Rz(c) in body sequence.
        a = math.radians(float(a_deg))
        b = math.radians(float(b_deg))
        c = math.radians(float(c_deg))

        ca, sa = math.cos(a), math.sin(a)
        cb, sb = math.cos(b), math.sin(b)
        cc, sc = math.cos(c), math.sin(c)

        Rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
        Ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
        Rz = np.array([[cc, -sc, 0.0], [sc, cc, 0.0], [0.0, 0.0, 1.0]])
        return Rx @ Ry @ Rz

    def cb_meca_lin_advance(self, msg: Float64MultiArray):
        """
        Receive a relative MoveLin advance command from the CURRENT TCP pose.

        Accepted formats:
            msg.data = [distance_mm]
            msg.data = [distance_mm, axis_column, axis_sign]
            msg.data = [distance_mm, axis_column, axis_sign, command_id]

        The 4-value format is duplicate-safe. This matters because the command
        is relative: executing the same message twice would advance twice.
        """
        if len(msg.data) not in (1, 3, 4):
            self.get_logger().error(
                'Need [distance_mm], [distance_mm, axis_column, axis_sign], or '
                '[distance_mm, axis_column, axis_sign, command_id], '
                f'received {len(msg.data)} values.'
            )
            return

        distance_mm = float(msg.data[0])
        axis_column = int(msg.data[1]) if len(msg.data) >= 2 else 2
        axis_sign = float(msg.data[2]) if len(msg.data) >= 3 else 1.0
        command_id = float(msg.data[3]) if len(msg.data) >= 4 else None

        if command_id is not None:
            if self.last_meca_lin_advance_command_id == command_id:
                self.get_logger().warn(
                    f'Ignoring duplicate /meca_lin_advance command_id={command_id:.0f}'
                )
                return
            self.last_meca_lin_advance_command_id = command_id
            self.last_meca_lin_advance_command_time = time.time()

        if axis_column not in (0, 1, 2):
            self.get_logger().error(f'axis_column must be 0, 1, or 2. Got {axis_column}.')
            return

        if axis_sign not in (-1.0, 1.0):
            self.get_logger().error(f'axis_sign must be +1 or -1. Got {axis_sign}.')
            return

        cmd_id_txt = (
            f', command_id={command_id:.0f}' if command_id is not None else ''
        )
        self.get_logger().info(
            'Received relative MoveLin advance: '
            f'distance={distance_mm:.3f} mm, axis_column={axis_column}, '
            f'axis_sign={axis_sign:+.0f}{cmd_id_txt}'
        )

        self.q.put(('cart_lin_advance_current', (distance_mm, axis_column, axis_sign)))

    def cb_pose_stamped(self, msg: PoseStamped):
        """
        Receive Cartesian target pose.

        IMPORTANT:
            This node assumes position is already in Meca500 native units: mm.

        If later you send PoseStamped directly from MoveIt, MoveIt usually
        uses metres. In that case you must multiply p.x, p.y, p.z by 1000
        before sending to MovePose/MoveLin.
        """

        p = msg.pose.position
        q = msg.pose.orientation

        roll, pitch, yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        goal = (
            float(p.x),
            float(p.y),
            float(p.z),
            math.degrees(roll),
            math.degrees(pitch),
            math.degrees(yaw),
        )

        self.get_logger().info(
            "Received Cartesian target for Meca500 TCP: "
            f"x={goal[0]:.3f} mm, "
            f"y={goal[1]:.3f} mm, "
            f"z={goal[2]:.3f} mm, "
            f"a={goal[3]:.3f} deg, "
            f"b={goal[4]:.3f} deg, "
            f"c={goal[5]:.3f} deg, "
            f"mode={self.cartesian_motion_mode}"
        )

        self.q.put(("cart", goal))

    # ─────────────────────────────────────────
    # FollowJointTrajectory callbacks
    # ─────────────────────────────────────────

    def goal_callback(self, goal_request):
        """
        Accept incoming FollowJointTrajectory goals from MoveIt.
        """
        self.get_logger().info("Received FollowJointTrajectory goal request.")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept trajectory cancellation requests.
        """
        self.get_logger().warn("Received request to cancel trajectory.")
        return CancelResponse.ACCEPT

    def densify_joint_waypoints(self, waypoints_deg):
        """
        Add intermediate joint waypoints if the jump between two consecutive
        waypoints is too large.

        Args:
            waypoints_deg:
                List of np.ndarray with shape (6,), in degrees.

        Returns:
            Dense list of np.ndarray with shape (6,), in degrees.
        """
        if len(waypoints_deg) <= 1:
            return waypoints_deg

        if self.max_joint_step_deg <= 0.0:
            return waypoints_deg

        dense_waypoints = [waypoints_deg[0]]

        for q_start, q_end in zip(waypoints_deg[:-1], waypoints_deg[1:]):
            delta = q_end - q_start
            max_delta = float(np.max(np.abs(delta)))

            n_steps = int(math.ceil(max_delta / self.max_joint_step_deg))

            if n_steps <= 1:
                dense_waypoints.append(q_end)
                continue

            for k in range(1, n_steps + 1):
                alpha = k / n_steps
                q_interp = q_start + alpha * delta
                dense_waypoints.append(q_interp)

        return dense_waypoints

    def move_joints_and_wait(self, q_deg):
        """
        Send one MoveJoints command and wait for completion.

        Default behaviour keeps WaitIdle() inside robot_lock because this is
        safest if mecademicpy.Robot is not thread-safe.

        If wait_idle_holds_lock is False, WaitIdle() is called outside the lock
        so monitor_loop can keep reading GetJoints()/GetPose() during motion.
        Use that mode only after confirming the robot API remains stable.
        """
        q_list = [float(x) for x in np.asarray(q_deg, dtype=float).reshape(6)]

        if self.wait_idle_holds_lock:
            with self.robot_lock:
                self.robot.MoveJoints(*q_list)
                self.robot.WaitIdle()
        else:
            with self.robot_lock:
                self.robot.MoveJoints(*q_list)

            # Experimental mode: lets monitor_loop update /joint_states while
            # the robot is moving, but may expose thread-safety issues in the
            # robot API on some setups.
            self.robot.WaitIdle()

    def meca_euler_xyz_deg_to_quat_xyzw(self, a, b, c):
        """
        Convert Meca500 Cartesian orientation to ROS quaternion.

        Meca500 reports alpha, beta, gamma as intrinsic/mobile XYZ Euler angles.
        tf_transformations.quaternion_from_euler() defaults to static/extrinsic
        XYZ, so using the default would give the wrong TCP orientation in TF.
        """
        if SciPyRotation is not None:
            q = SciPyRotation.from_euler(
                'XYZ',
                [float(a), float(b), float(c)],
                degrees=True,
            ).as_quat()  # [x, y, z, w]
            return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]

        # Fallback without scipy. In tf_transformations, 'rxyz' means rotating
        # axes, i.e. intrinsic XYZ.
        return quaternion_from_euler(
            math.radians(float(a)),
            math.radians(float(b)),
            math.radians(float(c)),
            axes='rxyz',
        )

    def decimate_joint_waypoints(self, waypoints_deg):
        """
        Reduce the number of waypoints executed by the real robot.

        MoveIt may output many small joint increments. This V1 bridge executes
        MoveJoints + WaitIdle for each accepted waypoint, so executing hundreds
        of tiny points can exceed MoveIt's execution timeout.

        The first and final waypoints are always kept.
        Intermediate waypoints are kept only if they differ from the previously
        kept waypoint by at least min_waypoint_delta_deg in any joint.
        """
        if len(waypoints_deg) <= 2:
            return waypoints_deg

        if self.min_waypoint_delta_deg <= 0.0:
            return waypoints_deg

        decimated = [waypoints_deg[0]]
        last_kept = waypoints_deg[0]

        for q in waypoints_deg[1:-1]:
            max_delta = float(np.max(np.abs(q - last_kept)))
            if max_delta >= self.min_waypoint_delta_deg:
                decimated.append(q)
                last_kept = q

        # Always keep the final target exactly.
        decimated.append(waypoints_deg[-1])

        return decimated

    def execute_trajectory_callback(self, goal_handle):
        """
        Execute a MoveIt trajectory on the real Meca500.

        MoveIt sends joint positions in radians.
        Meca500 MoveJoints expects degrees.

        V1 behaviour:
            - does not respect time_from_start
            - does not use blending
            - executes waypoint by waypoint
            - waits after each waypoint
            - slow but safe for first integration
            - follows the MoveIt planned path better than sending only
              the final joint target
        """

        trajectory = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()

        if len(trajectory.points) == 0:
            self.get_logger().error("Received empty trajectory.")
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "Empty trajectory."
            return result

        joint_names = list(trajectory.joint_names)

        try:
            order = [joint_names.index(j) for j in MECA_JOINT_NAMES]
        except ValueError as e:
            self.get_logger().error(
                f"Joint name mismatch. "
                f"Expected: {MECA_JOINT_NAMES}. "
                f"Received: {joint_names}. "
                f"Error: {e}"
            )
            goal_handle.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            result.error_string = "Joint name mismatch."
            return result

        try:
            # Low speed for first real-robot integration.
            with self.robot_lock:
                self.robot.SetJointVel(self.joint_vel_percent)

                try:
                    self.robot.SetJointAcc(self.joint_acc_percent)
                except Exception as e:
                    self.get_logger().warn(f"SetJointAcc failed: {e}")

            # 1. Convert MoveIt waypoints from radians to Meca500 degrees.
            waypoints_deg = []

            for point_idx, point in enumerate(trajectory.points):
                if len(point.positions) < 6:
                    self.get_logger().error(
                        f"Trajectory point {point_idx} has fewer than 6 positions."
                    )
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                    result.error_string = (
                        f"Trajectory point {point_idx} has fewer than 6 positions."
                    )
                    return result

                q_rad_ordered = np.array(
                    [point.positions[i] for i in order],
                    dtype=float,
                )

                q_deg = np.rad2deg(q_rad_ordered)
                waypoints_deg.append(q_deg)

            original_count = len(waypoints_deg)

            # 2. Densify waypoints to avoid large joint jumps.
            waypoints_deg = self.densify_joint_waypoints(waypoints_deg)
            dense_count = len(waypoints_deg)

            # 3. Decimate very small increments, because this V1 executor waits
            # after each MoveJoints command. This avoids MoveIt execution timeout.
            waypoints_deg = self.decimate_joint_waypoints(waypoints_deg)
            decimated_count = len(waypoints_deg)

            self.get_logger().info(
                f"Trajectory waypoints: {original_count} original, "
                f"{dense_count} after densification, "
                f"{decimated_count} after decimation "
                f"(max step = {self.max_joint_step_deg:.2f} deg, "
                f"min delta = {self.min_waypoint_delta_deg:.2f} deg)."
            )

            # 3. Execute waypoint by waypoint on the real robot.
            previous_q_deg = None

            for idx, q_deg in enumerate(waypoints_deg):
                if goal_handle.is_cancel_requested:
                    self.get_logger().warn("Trajectory execution cancelled.")
                    goal_handle.canceled()
                    result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                    result.error_string = "Trajectory cancelled by request."
                    return result

                # Optional: skip almost identical points.
                if previous_q_deg is not None:
                    if np.max(np.abs(q_deg - previous_q_deg)) < 0.05:
                        continue

                self.get_logger().info(
                    f"Moving waypoint {idx + 1}/{len(waypoints_deg)} "
                    f"to joints deg: {np.round(q_deg, 3).tolist()}"
                )

                self.move_joints_and_wait(q_deg)

                previous_q_deg = q_deg

            # ── Verificación de posición final ────────────────────────────────
            # Comprueba que el robot ha llegado realmente al último waypoint.
            # Si MoveJoints/WaitIdle devolvió OK pero el robot se quedó corto
            # (límite de workspace silencioso, error interno del controlador...),
            # esto bloquea que MoveIt reporte SUCCESS falsamente.
            if self.final_joint_tolerance_deg > 0.0 and len(waypoints_deg) > 0:
                try:
                    with self.robot_lock:
                        actual_joints = np.array(
                            self.robot.GetJoints(), dtype=float
                        )
                    target_joints = np.asarray(waypoints_deg[-1], dtype=float)
                    errors_deg = np.abs(actual_joints - target_joints)
                    max_error_deg = float(np.max(errors_deg))
                    worst_joint = int(np.argmax(errors_deg)) + 1

                    if max_error_deg > self.final_joint_tolerance_deg:
                        self.get_logger().error(
                            f"Final joint position error too large: "
                            f"J{worst_joint} error = {max_error_deg:.2f} deg "
                            f"(tolerance = {self.final_joint_tolerance_deg:.2f} deg). "
                            f"Robot may not have reached T_stage."
                        )
                        self.get_logger().error(
                            f"  Target: {np.round(target_joints, 2).tolist()}"
                        )
                        self.get_logger().error(
                            f"  Actual: {np.round(actual_joints, 2).tolist()}"
                        )
                        goal_handle.abort()
                        result.error_code = (
                            FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                        )
                        result.error_string = (
                            f"Final joint error {max_error_deg:.2f} deg on J{worst_joint} "
                            f"exceeds tolerance {self.final_joint_tolerance_deg:.2f} deg."
                        )
                        return result

                    self.get_logger().info(
                        f"Final joint position verified: "
                        f"max error = {max_error_deg:.3f} deg on J{worst_joint} "
                        f"(tolerance = {self.final_joint_tolerance_deg:.2f} deg). OK."
                    )
                except Exception as verify_exc:
                    self.get_logger().error(
                        f"Could not verify final joint position: {verify_exc}. "
                        "Aborting trajectory to avoid false MoveIt success."
                    )
                    goal_handle.abort()
                    result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
                    result.error_string = (
                        f"Could not verify final joint position: {verify_exc}"
                    )
                    return result

            goal_handle.succeed()
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            result.error_string = "Trajectory executed successfully."
            self.get_logger().info("Trajectory execution completed.")
            return result

        except Exception as e:
            self.get_logger().error(f"Trajectory execution failed: {e}")
            goal_handle.abort()
            result.error_code = (
                FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            )
            result.error_string = str(e)
            return result

    # ─────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────

    def monitor_loop(self):
        """
        Continuously read robot pose and joints.

        After SetTrf, GetPose is the pose of the active TRF/TCP.
        Therefore, with SetTrf(0, 0, L, 0, 0, 0), GetPose corresponds to
        needle_tcp.
        """
        while rclpy.ok():
            try:
                with self.robot_lock:
                    x, y, z, a, b, c = self.robot.GetPose()
                    joints = self.robot.GetJoints()

                with self.latest_pose_lock:
                    self.latest_pose = (x, y, z, a, b, c)

                with self.latest_joints_lock:
                    self.latest_joints = list(joints)

            except Exception as e:
                self.get_logger().error(f'Monitor error: {e}')

            time.sleep(self.tf_period)

    # ─────────────────────────────────────────
    # State publishing
    # ─────────────────────────────────────────

    def publish_state(self):
        self.publish_tf()
        self.publish_joint_states()

    def publish_tf(self):
        """
        Publish TF from robot base to TCP/TRF.

        This publishes the pose returned by Meca500 GetPose.
        Because SetTrf is active, this TF is:
            meca_base_link -> needle_tcp

        Meca500 returns position in mm.
        ROS TF uses metres.
        """
        with self.latest_pose_lock:
            pose = self.latest_pose

        if pose is None:
            return

        x, y, z, a, b, c = pose

        qx, qy, qz, qw = self.meca_euler_xyz_deg_to_quat_xyzw(a, b, c)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.tf_parent_frame
        t.child_frame_id = self.tf_child_frame

        # Meca500 mm -> ROS metres
        t.transform.translation.x = float(x) / 1000.0
        t.transform.translation.y = float(y) / 1000.0
        t.transform.translation.z = float(z) / 1000.0

        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)

    def publish_joint_states(self):
        """
        Publish joint states in ROS standard units: radians.

        Meca500 API:
            degrees

        ROS / MoveIt:
            radians
        """
        with self.latest_joints_lock:
            joints = self.latest_joints

        if joints is None:
            return

        q_deg = np.array(joints, dtype=float)
        q_rad = np.deg2rad(q_deg)

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = MECA_JOINT_NAMES
        msg.position = q_rad.tolist()

        self.joint_pub.publish(msg)

    # ─────────────────────────────────────────
    # Motion worker for legacy topics
    # ─────────────────────────────────────────

    def run(self):
        """
        Execute queued commands sequentially.

        This worker is kept for compatibility with your previous workflow:

            /joint_targets   -> MoveJoints
            /pose_for_meca   -> MovePose or MoveLin
            /meca_lin_pose   -> MoveLin with native [x,y,z,a,b,c]
            /meca_move_pose  -> MovePose with native [x,y,z,a,b,c]

        Joint commands:
            MoveJoints(j1, ..., j6)

        Cartesian commands:
            MovePose(x, y, z, a, b, c) if cartesian_motion_mode='pose'
            MoveLin(x, y, z, a, b, c)  if cartesian_motion_mode='lin'

        Because SetTrf is active, Cartesian targets refer to the needle TCP.
        """
        while rclpy.ok():
            cmd_type, vals = self.q.get()

            try:
                with self.robot_lock:
                    if cmd_type == "joint":
                        self.get_logger().info(
                            f'Executing MoveJoints: {vals}'
                        )
                        self.robot.MoveJoints(*vals)

                    elif cmd_type == "cart":
                        if self.cartesian_motion_mode == "lin":
                            self.get_logger().info(
                                f'Executing MoveLin to needle TCP target: {vals}'
                            )
                            self.robot.MoveLin(*vals)
                        else:
                            self.get_logger().info(
                                f'Executing MovePose to needle TCP target: {vals}'
                            )
                            self.robot.MovePose(*vals)

                    elif cmd_type == "cart_lin":
                        self.get_logger().info(
                            f'Executing native MoveLin to needle TCP target: {vals}'
                        )
                        self.robot.MoveLin(*vals)

                    elif cmd_type in ("cart_pose", "move_pose"):
                        self.get_logger().info(
                            f'Executing native MovePose to needle TCP target: {vals}'
                        )
                        self.robot.MovePose(*vals)

                    elif cmd_type == "cart_lin_advance_current":
                        distance_mm, axis_column, axis_sign = vals

                        # Defensive ResumeMotion before GetPose.
                        # If the preceding MovePose raised an exception (workspace
                        # limit, controller error, etc.), the robot may be in an
                        # error state.  ResumeMotion is a no-op when the robot is
                        # already idle and healthy, so this is always safe to call.
                        try:
                            self.robot.ResumeMotion()
                        except Exception as _rm_exc:
                            self.get_logger().warn(
                                f'ResumeMotion before GetPose (defensive): {_rm_exc}'
                            )

                        # Read the real current TCP pose from the Meca500.
                        # GetPose() uses the active TRF/TCP and Meca500 native units.
                        x, y, z, a, b, c = self.robot.GetPose()

                        # ── Diagnostic: log joint angles to detect workspace limits ──
                        try:
                            joints_diag = self.robot.GetJoints()
                            self.get_logger().info(
                                'Joints before advance (deg): '
                                + str([round(j, 2) for j in joints_diag])
                                + '  limits: ax2=±70, ax3=[-135,+70], ax5=±115'
                            )
                        except Exception as _e:
                            self.get_logger().warn(f'GetJoints diagnostic failed: {_e}')

                        R_tcp = self._meca_intrinsic_xyz_to_matrix(a, b, c)

                        # ── Diagnostic: log all 3 TCP axes in base frame ──
                        # Compare with the expected insertion direction to confirm
                        # which axis_column to use for this physical setup.
                        #   Meca500 FRF convention: +Z is the tool approach axis
                        #   (consistent with SetTrf(0,0,L,0,0,0)).
                        #   → use axis_column=2 for insertion advance.
                        #   The URDF/MoveIt tcp_link uses +X as approach, but
                        #   GetPose() reports the real FRF orientation, not tcp_link.
                        x_ax = R_tcp[:, 0]
                        y_ax = R_tcp[:, 1]
                        z_ax = R_tcp[:, 2]
                        self.get_logger().info(
                            'TCP axes in base frame — '
                            f'X=[{x_ax[0]:.3f},{x_ax[1]:.3f},{x_ax[2]:.3f}]  '
                            f'Y=[{y_ax[0]:.3f},{y_ax[1]:.3f},{y_ax[2]:.3f}]  '
                            f'Z=[{z_ax[0]:.3f},{z_ax[1]:.3f},{z_ax[2]:.3f}]  '
                            f'→ using column={axis_column}'
                        )

                        axis = np.asarray(R_tcp[:, axis_column], dtype=np.float64)
                        axis_norm = np.linalg.norm(axis)
                        if axis_norm < 1e-9:
                            raise RuntimeError('Current TCP axis has near-zero norm')
                        axis = axis / axis_norm

                        dx, dy, dz = axis_sign * distance_mm * axis
                        target = (
                            float(x + dx),
                            float(y + dy),
                            float(z + dz),
                            float(a),
                            float(b),
                            float(c),
                        )

                        self.get_logger().info(
                            'Current TCP pose before advance: '
                            f'x={x:.3f}, y={y:.3f}, z={z:.3f}, '
                            f'a={a:.3f}, b={b:.3f}, c={c:.3f}'
                        )
                        self.get_logger().info(
                            'Advance axis in base frame: '
                            f'[{axis[0]:.6f}, {axis[1]:.6f}, {axis[2]:.6f}] '
                            f'(column={axis_column}, sign={axis_sign:+.0f}, distance={distance_mm:.3f} mm)'
                        )
                        self.get_logger().info(
                            f'Executing relative MoveLin advance to target: {target}'
                        )
                        self.robot.MoveLin(*target)

                    if self.wait_idle_holds_lock:
                        self.robot.WaitIdle()

                if not self.wait_idle_holds_lock:
                    self.robot.WaitIdle()

                self.get_logger().info('Motion completed.')

            except Exception as e:
                self.get_logger().error(f'Motion error: {e}')

            finally:
                self.q.task_done()


def main(args=None):
    rclpy.init(args=args)

    node = JointCommandNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
