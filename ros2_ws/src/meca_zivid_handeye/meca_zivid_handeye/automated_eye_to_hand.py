#!/usr/bin/env python3

import json
import math
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import rclpy
from geometry_msgs.msg import Point, Pose
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from zivid_interfaces.msg import DetectionResultCalibrationBoard, HandEyeCalibrationObjects
from zivid_interfaces.srv import (
    HandEyeCalibrationCalibrate,
    HandEyeCalibrationCapture,
    HandEyeCalibrationStart,
)

from mecademicpy.robot import Robot


DEFAULT_JOINT_CONFIGURATIONS_DEG = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
]

DEFAULT_SETTINGS_YAML = """
__version__:
  serializer: 1
  data: 22
Settings:
  Acquisitions:
    - Acquisition:
        Aperture: 5.66
        ExposureTime: 8333
  Processing:
    Filters:
      Outlier:
        Removal:
          Enabled: yes
          Threshold: 5
"""


@dataclass
class CaptureCandidate:
    index: int
    capture_handle: int
    robot_pose: Pose
    board_pose: Pose
    board_centroid: Point
    record: dict


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_normalize(q):
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def quat_from_axis_angle(axis, angle_rad: float):
    ax, ay, az = axis
    half_angle = 0.5 * angle_rad
    sin_half = math.sin(half_angle)
    return (math.cos(half_angle), ax * sin_half, ay * sin_half, az * sin_half)


def mecademic_intrinsic_xyz_to_quat_xyzw(alpha_deg: float, beta_deg: float, gamma_deg: float):
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(alpha_deg))
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(beta_deg))
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(gamma_deg))
    qw, qx_out, qy_out, qz_out = quat_normalize(quat_mul(quat_mul(qx, qy), qz))
    return (qx_out, qy_out, qz_out, qw)


def make_pose_from_meca_pose_meters(meca_pose) -> Pose:
    x_mm, y_mm, z_mm, alpha_deg, beta_deg, gamma_deg = [float(value) for value in meca_pose]
    qx, qy, qz, qw = mecademic_intrinsic_xyz_to_quat_xyzw(alpha_deg, beta_deg, gamma_deg)

    pose = Pose()
    pose.position.x = x_mm / 1000.0
    pose.position.y = y_mm / 1000.0
    pose.position.z = z_mm / 1000.0
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def pose_to_dict(pose: Pose) -> dict:
    return {
        'position_m': {
            'x': float(pose.position.x),
            'y': float(pose.position.y),
            'z': float(pose.position.z),
        },
        'orientation_xyzw': {
            'x': float(pose.orientation.x),
            'y': float(pose.orientation.y),
            'z': float(pose.orientation.z),
            'w': float(pose.orientation.w),
        },
    }


def transform_to_dict(transform) -> dict:
    return {
        'translation_m': {
            'x': float(transform.translation.x),
            'y': float(transform.translation.y),
            'z': float(transform.translation.z),
        },
        'rotation_xyzw': {
            'x': float(transform.rotation.x),
            'y': float(transform.rotation.y),
            'z': float(transform.rotation.z),
            'w': float(transform.rotation.w),
        },
    }


def quaternion_angle_deg(q1, q2) -> float:
    dot = abs(
        float(q1.x) * float(q2.x)
        + float(q1.y) * float(q2.y)
        + float(q1.z) * float(q2.z)
        + float(q1.w) * float(q2.w)
    )
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


def pose_delta(candidate: Pose, reference: Pose) -> tuple[float, float]:
    dx = float(candidate.position.x) - float(reference.position.x)
    dy = float(candidate.position.y) - float(reference.position.y)
    dz = float(candidate.position.z) - float(reference.position.z)
    translation_m = math.sqrt(dx * dx + dy * dy + dz * dz)
    rotation_deg = quaternion_angle_deg(candidate.orientation, reference.orientation)
    return translation_m, rotation_deg


def point_distance_m(candidate: Point, reference: Point) -> float:
    dx = float(candidate.x) - float(reference.x)
    dy = float(candidate.y) - float(reference.y)
    dz = float(candidate.z) - float(reference.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def copy_pose(source: Pose) -> Pose:
    pose = Pose()
    pose.position.x = float(source.position.x)
    pose.position.y = float(source.position.y)
    pose.position.z = float(source.position.z)
    pose.orientation.x = float(source.orientation.x)
    pose.orientation.y = float(source.orientation.y)
    pose.orientation.z = float(source.orientation.z)
    pose.orientation.w = float(source.orientation.w)
    return pose


def copy_point(source: Point) -> Point:
    point = Point()
    point.x = float(source.x)
    point.y = float(source.y)
    point.z = float(source.z)
    return point


def validate_joint_configurations(joint_configurations: Iterable[Iterable[float]]):
    validated = []
    for index, joints in enumerate(joint_configurations):
        values = [float(value) for value in joints]
        if len(values) != 6:
            raise ValueError(
                f'Joint configuration {index} must contain exactly 6 values, got {len(values)}'
            )
        validated.append(values)
    if not validated:
        raise ValueError('At least one joint configuration is required')
    return validated


def load_joint_configurations(path: str):
    if not path:
        return DEFAULT_JOINT_CONFIGURATIONS_DEG

    with open(Path(path).expanduser(), 'r', encoding='utf-8') as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get('joint_configurations_deg')
    if data is None:
        raise ValueError(
            'Joint configuration file must contain a "joint_configurations_deg" list'
        )
    return data


def unique_working_directory(base_dir: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return str(Path(base_dir) / f'meca_zivid_handeye_{timestamp}')


class KeyboardSkipController:
    def __init__(self, node):
        self.node = node
        self.enabled = bool(sys.stdin.isatty())
        self.skip_requested = threading.Event()
        self.finish_requested = threading.Event()
        self.shutdown_requested = threading.Event()
        self.thread = None
        self.original_terminal_settings = None

    def start(self) -> None:
        if not self.enabled:
            self.node.get_logger().warn(
                'Keyboard pose skipping disabled because stdin is not a TTY.'
            )
            return

        try:
            self.original_terminal_settings = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())
        except Exception as error:
            self.enabled = False
            self.node.get_logger().warn(f'Keyboard pose skipping disabled: {error}')
            return

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        self.node.get_logger().info(
            'Keyboard controls enabled: SPACE skips current pose, ENTER finishes '
            'candidate collection.'
        )

    def stop(self) -> None:
        self.shutdown_requested.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.original_terminal_settings is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(),
                    termios.TCSADRAIN,
                    self.original_terminal_settings,
                )
            except Exception as error:
                self.node.get_logger().warn(f'Failed to restore terminal settings: {error}')

    def clear(self) -> None:
        self.skip_requested.clear()

    def requested(self) -> bool:
        return self.skip_requested.is_set()

    def finish_collection_requested(self) -> bool:
        return self.finish_requested.is_set()

    def _run(self) -> None:
        while not self.shutdown_requested.is_set():
            try:
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            except Exception:
                continue
            if not readable:
                continue

            char = sys.stdin.read(1)
            if char == ' ':
                self.skip_requested.set()
                self.node.on_keyboard_skip_requested()
            elif char in ('\n', '\r'):
                self.finish_requested.set()
                self.node.on_keyboard_finish_requested()


class AutomatedEyeToHand(Node):
    def __init__(self):
        super().__init__('automated_eye_to_hand')

        self.declare_parameter('robot_ip', '192.168.0.100')
        self.declare_parameter('joint_configurations_file', '')
        self.declare_parameter('joint_velocity_deg_s', 2.0)
        self.declare_parameter('reset_error', True)
        self.declare_parameter('resume_motion', True)
        self.declare_parameter('activate_robot', False)
        self.declare_parameter('home_robot', False)
        self.declare_parameter('wait_idle', True)
        self.declare_parameter('settle_s', 1.0)
        self.declare_parameter('move_robot', True)
        self.declare_parameter('enable_keyboard_skip', True)
        self.declare_parameter('pause_motion_on_keyboard_skip', True)
        self.declare_parameter('resume_motion_after_keyboard_skip', True)
        self.declare_parameter('reconnect_after_keyboard_skip', True)
        self.declare_parameter('skip_duplicate_robot_poses', True)
        self.declare_parameter('duplicate_position_tolerance_m', 0.0001)
        self.declare_parameter('duplicate_rotation_tolerance_deg', 0.01)
        self.declare_parameter('min_successful_captures', 6)
        self.declare_parameter('max_successful_captures', 0)
        self.declare_parameter('selected_capture_count', 20)
        self.declare_parameter('robot_position_scale_m', 0.05)
        self.declare_parameter('robot_rotation_scale_deg', 10.0)
        self.declare_parameter('board_position_scale_m', 0.05)
        self.declare_parameter('board_rotation_scale_deg', 10.0)
        self.declare_parameter('robot_position_weight', 1.0)
        self.declare_parameter('robot_rotation_weight', 1.0)
        self.declare_parameter('board_position_weight', 1.0)
        self.declare_parameter('board_rotation_weight', 1.0)

        self.declare_parameter('working_directory', '')
        self.declare_parameter('working_directory_base', '/tmp')
        self.declare_parameter('result_json_filename', 'hand_eye_result_eye_to_hand.json')
        self.declare_parameter('legacy_tf_yaml_filename', 'tf2_zivid_robotBase.generated.yaml')
        self.declare_parameter('legacy_tf_parent_frame', 'meca_base_link')
        self.declare_parameter('legacy_tf_child_frame', 'pcl_frame')

        self.declare_parameter('start_service', '/hand_eye_calibration/start')
        self.declare_parameter('capture_service', '/hand_eye_calibration/capture')
        self.declare_parameter('calibrate_service', '/hand_eye_calibration/calibrate')
        self.declare_parameter('zivid_parameter_node', '/zivid_camera')
        self.declare_parameter('set_color_space_srgb', True)
        self.declare_parameter('set_settings_yaml', True)
        self.declare_parameter('settings_yaml', DEFAULT_SETTINGS_YAML)
        self.declare_parameter('service_wait_timeout_s', 180.0)
        self.declare_parameter('service_call_timeout_s', 180.0)

        self.robot = None
        self.keyboard_skip = KeyboardSkipController(self)
        self.motion_paused_by_keyboard_skip = False

    def get_bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def get_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def get_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def get_string(self, name: str) -> str:
        return str(self.get_parameter(name).value)

    def _optional_robot_call(self, label: str, fn: Callable, *args, **kwargs) -> None:
        try:
            fn(*args, **kwargs)
        except Exception as error:
            self.get_logger().warn(f'{label} failed; continuing: {error}')

    def _required_robot_call(self, label: str, fn: Callable, *args, **kwargs) -> bool:
        try:
            fn(*args, **kwargs)
            return True
        except Exception as error:
            self.get_logger().error(f'{label} failed: {error}')
            return False

    def on_keyboard_skip_requested(self) -> None:
        self.get_logger().warn('SPACE pressed: current pose will be skipped.')
        if (
            self.robot is not None
            and self.get_bool('pause_motion_on_keyboard_skip')
        ):
            self.pause_robot_motion_for_skip()

    def on_keyboard_finish_requested(self) -> None:
        self.get_logger().warn(
            'ENTER pressed: candidate collection will stop after the current pose.'
        )
        if (
            self.robot is not None
            and self.get_bool('pause_motion_on_keyboard_skip')
        ):
            self.pause_robot_motion_for_skip()

    def pause_robot_motion_for_skip(self) -> None:
        for method_name in ('PauseMotion', 'ClearMotion'):
            method = getattr(self.robot, method_name, None)
            if method is None:
                continue
            try:
                method()
                self.motion_paused_by_keyboard_skip = True
                self.get_logger().warn(f'{method_name} sent after keyboard skip.')
            except Exception as error:
                self.get_logger().warn(f'{method_name} after keyboard skip failed: {error}')

    def resume_after_keyboard_skip_if_needed(self) -> None:
        if not self.motion_paused_by_keyboard_skip:
            return
        if not self.get_bool('resume_motion_after_keyboard_skip'):
            return

        method = getattr(self.robot, 'ResumeMotion', None)
        if method is None:
            self.motion_paused_by_keyboard_skip = False
            return
        try:
            method()
            self.get_logger().info('ResumeMotion sent after skipped pose.')
        except Exception as error:
            self.get_logger().warn(f'ResumeMotion after skipped pose failed: {error}')
        finally:
            self.motion_paused_by_keyboard_skip = False

    def connect_robot(self) -> bool:
        robot_ip = self.get_string('robot_ip')
        self.get_logger().info(f'Connecting to Mecademic robot at {robot_ip}...')

        try:
            self.robot = self.create_robot()
        except Exception as error:
            self.get_logger().error(f'Failed to create Robot object: {error}')
            return False

        return self.initialize_robot_connection()

    def create_robot(self):
        try:
            return Robot(disconnect_on_exception=False)
        except TypeError:
            return Robot()

    def initialize_robot_connection(self) -> bool:
        robot_ip = self.get_string('robot_ip')
        if not self._required_robot_call('Connect', self.robot.Connect, address=robot_ip):
            return False

        if self.get_bool('reset_error'):
            self._optional_robot_call('ResetError', self.robot.ResetError)
        if self.get_bool('resume_motion'):
            self._optional_robot_call('ResumeMotion', self.robot.ResumeMotion)
        if self.get_bool('activate_robot'):
            self._optional_robot_call('ActivateRobot', self.robot.ActivateRobot)
        if self.get_bool('home_robot'):
            self._optional_robot_call('Home', self.robot.Home)
            if self.get_bool('wait_idle'):
                self._optional_robot_call('WaitIdle after Home', self.robot.WaitIdle)

        joint_velocity = self.get_float('joint_velocity_deg_s')
        if joint_velocity > 0.0:
            self._optional_robot_call('SetJointVel', self.robot.SetJointVel, joint_velocity)

        return True

    def reconnect_robot_after_skip(self) -> None:
        if not self.get_bool('reconnect_after_keyboard_skip'):
            return

        try:
            if self.robot is not None and self.robot.IsConnected():
                return
        except Exception:
            pass

        self.get_logger().warn('Robot appears disconnected after skip; reconnecting.')
        try:
            if self.robot is not None:
                self._optional_robot_call('Disconnect before reconnect', self.robot.Disconnect)
            self.robot = self.create_robot()
            if self.initialize_robot_connection():
                self.get_logger().info('Robot reconnected after skipped pose.')
            else:
                self.get_logger().error('Robot reconnect after skipped pose failed.')
        except Exception as error:
            self.get_logger().error(f'Robot reconnect after skipped pose failed: {error}')

    def recover_after_keyboard_skip(self) -> None:
        self.resume_after_keyboard_skip_if_needed()
        self.reconnect_robot_after_skip()
        self.keyboard_skip.clear()

    def disconnect_robot(self) -> None:
        if self.robot is not None:
            self._optional_robot_call('Disconnect', self.robot.Disconnect)

    def wait_for_client(self, client, label: str) -> bool:
        timeout_s = self.get_float('service_wait_timeout_s')
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not client.wait_for_service(timeout_sec=3.0):
            if time.monotonic() >= deadline:
                self.get_logger().error(f'Timed out waiting for {label}')
                return False
            self.get_logger().info(f'Waiting for {label}...')
        return True

    def call_service(self, client, request, label: str):
        future = client.call_async(request)
        timeout_s = self.get_float('service_call_timeout_s')
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done():
            raise RuntimeError(f'Timed out or failed while calling {label}')
        if future.exception() is not None:
            raise RuntimeError(f'Failed while calling {label}: {future.exception()}')
        return future.result()

    def set_zivid_parameter(self, name: str, value) -> bool:
        parameter_node = self.get_string('zivid_parameter_node')
        service_name = parameter_node.rstrip('/') + '/set_parameters'
        client = self.create_client(SetParameters, service_name)
        if not self.wait_for_client(client, service_name):
            return False

        request = SetParameters.Request()
        request.parameters = [Parameter(name, value=value).to_parameter_msg()]
        response = self.call_service(client, request, service_name)

        if not response.results:
            self.get_logger().error(f'No response while setting Zivid parameter {name}')
            return False
        if not response.results[0].successful:
            self.get_logger().error(
                f'Failed to set Zivid parameter {name}: {response.results[0].reason}'
            )
            return False
        return True

    def configure_zivid(self) -> bool:
        if self.get_bool('set_color_space_srgb'):
            self.get_logger().info('Setting Zivid color_space to srgb')
            if not self.set_zivid_parameter('color_space', 'srgb'):
                return False

        if self.get_bool('set_settings_yaml'):
            self.get_logger().info('Setting Zivid settings_yaml for hand-eye captures')
            if not self.set_zivid_parameter('settings_yaml', self.get_string('settings_yaml')):
                return False

        return True

    def start_hand_eye_session(self, working_directory: str) -> bool:
        client = self.create_client(
            HandEyeCalibrationStart,
            self.get_string('start_service'),
        )
        if not self.wait_for_client(client, self.get_string('start_service')):
            return False

        request = HandEyeCalibrationStart.Request()
        request.working_directory = working_directory
        request.calibration_objects.type = HandEyeCalibrationObjects.CALIBRATION_BOARD

        self.get_logger().info(f'Starting Zivid eye-to-hand session in {working_directory}')
        response = self.call_service(client, request, self.get_string('start_service'))
        if not response.success:
            self.get_logger().error(f'Hand-eye start failed: {response.message}')
            return False
        return True

    def capture(self, robot_pose: Pose):
        client = self.create_client(
            HandEyeCalibrationCapture,
            self.get_string('capture_service'),
        )
        if not self.wait_for_client(client, self.get_string('capture_service')):
            return None

        request = HandEyeCalibrationCapture.Request()
        request.robot_pose = robot_pose
        return self.call_service(client, request, self.get_string('capture_service'))

    def calibrate(self, capture_handles: list[int]):
        client = self.create_client(
            HandEyeCalibrationCalibrate,
            self.get_string('calibrate_service'),
        )
        if not self.wait_for_client(client, self.get_string('calibrate_service')):
            return None

        request = HandEyeCalibrationCalibrate.Request()
        request.configuration = HandEyeCalibrationCalibrate.Request.EYE_TO_HAND
        request.capture_handles = capture_handles
        return self.call_service(client, request, self.get_string('calibrate_service'))

    def get_robot_pose(self):
        try:
            return self.robot.GetPose()
        except TypeError:
            return self.robot.GetPose(synchronous_update=False)

    def move_to_joints(self, joints_deg: list[float]) -> str:
        if self.keyboard_skip.requested():
            return 'skip'
        if self.keyboard_skip.finish_collection_requested():
            return 'finish'

        self.resume_after_keyboard_skip_if_needed()
        self.get_logger().info(
            'Moving robot to joints [deg]: '
            + ', '.join(f'{value:.3f}' for value in joints_deg)
        )
        if not self._required_robot_call('MoveJoints', self.robot.MoveJoints, *joints_deg):
            if self.keyboard_skip.finish_collection_requested():
                self.recover_after_keyboard_skip()
                return 'finish'
            if self.keyboard_skip.requested():
                self.recover_after_keyboard_skip()
                return 'skip'
            return 'failed'
        if self.keyboard_skip.requested():
            return 'skip'
        if self.keyboard_skip.finish_collection_requested():
            return 'finish'

        if self.get_bool('wait_idle'):
            if not self._required_robot_call('WaitIdle after MoveJoints', self.robot.WaitIdle):
                if self.keyboard_skip.finish_collection_requested():
                    self.recover_after_keyboard_skip()
                    return 'finish'
                if self.keyboard_skip.requested():
                    self.recover_after_keyboard_skip()
                    return 'skip'
                return 'failed'
        if self.keyboard_skip.requested():
            self.recover_after_keyboard_skip()
            return 'skip'
        if self.keyboard_skip.finish_collection_requested():
            self.recover_after_keyboard_skip()
            return 'finish'

        settle_s = self.get_float('settle_s')
        if settle_s > 0.0:
            self.get_logger().info(f'Waiting {settle_s:.2f}s for settling')
            deadline = time.monotonic() + settle_s
            while time.monotonic() < deadline:
                if self.keyboard_skip.requested():
                    self.recover_after_keyboard_skip()
                    return 'skip'
                if self.keyboard_skip.finish_collection_requested():
                    self.recover_after_keyboard_skip()
                    return 'finish'
                time.sleep(min(0.05, deadline - time.monotonic()))
        return 'ok'

    def make_working_directory(self) -> str:
        configured = self.get_string('working_directory').strip()
        if configured:
            working_directory = Path(configured).expanduser()
        else:
            working_directory = Path(
                unique_working_directory(self.get_string('working_directory_base'))
            )

        if not working_directory.is_absolute():
            raise ValueError(f'working_directory must be absolute: {working_directory}')

        working_directory.mkdir(parents=True, exist_ok=True)
        if any(working_directory.iterdir()):
            raise ValueError(
                f'working_directory must be empty for a new Zivid session: {working_directory}'
            )
        return str(working_directory)

    def warn_if_duplicate_poses(self, joint_configurations: list[list[float]]) -> None:
        rounded = {tuple(round(value, 5) for value in joints) for joints in joint_configurations}
        if len(rounded) != len(joint_configurations):
            self.get_logger().warn(
                'Some joint configurations are duplicates. The placeholder all-zero list is '
                'expected to fail final calibration until you replace it with distinct poses.'
            )

    def duplicate_robot_pose_index(
        self,
        candidate: Pose,
        accepted_robot_poses: list[Pose],
    ) -> int | None:
        if not self.get_bool('skip_duplicate_robot_poses'):
            return None

        position_tolerance_m = self.get_float('duplicate_position_tolerance_m')
        rotation_tolerance_deg = self.get_float('duplicate_rotation_tolerance_deg')
        for index, reference in enumerate(accepted_robot_poses):
            translation_m, rotation_deg = pose_delta(candidate, reference)
            if translation_m <= position_tolerance_m and rotation_deg <= rotation_tolerance_deg:
                return index
        return None

    def operator_skip_record(
        self,
        index: int,
        joints_deg: list[float],
        meca_pose=None,
        robot_pose: Pose | None = None,
    ) -> dict:
        record = {
            'index': index,
            'commanded_joints_deg': joints_deg,
            'success': False,
            'candidate': False,
            'accepted': False,
            'selected_for_calibration': False,
            'capture_handle': -1,
            'skipped': True,
            'skip_reason': 'operator_keyboard_skip',
        }
        if meca_pose is not None:
            record['meca_pose_mm_deg'] = meca_pose
        if robot_pose is not None:
            record['robot_pose_ros_m'] = pose_to_dict(robot_pose)
        return record

    def candidate_distance(
        self,
        candidate: CaptureCandidate,
        reference: CaptureCandidate,
    ) -> float:
        robot_translation_m, robot_rotation_deg = pose_delta(
            candidate.robot_pose,
            reference.robot_pose,
        )
        board_translation_m = point_distance_m(
            candidate.board_centroid,
            reference.board_centroid,
        )
        board_rotation_deg = quaternion_angle_deg(
            candidate.board_pose.orientation,
            reference.board_pose.orientation,
        )

        robot_position = (
            robot_translation_m
            / max(1e-9, self.get_float('robot_position_scale_m'))
        )
        robot_rotation = (
            robot_rotation_deg
            / max(1e-9, self.get_float('robot_rotation_scale_deg'))
        )
        board_position = (
            board_translation_m
            / max(1e-9, self.get_float('board_position_scale_m'))
        )
        board_rotation = (
            board_rotation_deg
            / max(1e-9, self.get_float('board_rotation_scale_deg'))
        )

        weighted_terms = [
            self.get_float('robot_position_weight') * robot_position,
            self.get_float('robot_rotation_weight') * robot_rotation,
            self.get_float('board_position_weight') * board_position,
            self.get_float('board_rotation_weight') * board_rotation,
        ]
        return math.sqrt(sum(term * term for term in weighted_terms))

    def select_calibration_candidates(
        self,
        candidates: list[CaptureCandidate],
        selected_capture_count: int,
    ) -> list[CaptureCandidate]:
        for candidate in candidates:
            candidate.record['selected_for_calibration'] = False
            candidate.record['selection_order'] = None
            candidate.record['selection_distance_score'] = None

        if selected_capture_count == 0 or len(candidates) <= selected_capture_count:
            for order, candidate in enumerate(candidates, start=1):
                candidate.record['selected_for_calibration'] = True
                candidate.record['selection_order'] = order
                candidate.record['selection_distance_score'] = 0.0
            return list(candidates)

        best_pair = (0, 1)
        best_distance = -1.0
        for first_index in range(len(candidates)):
            for second_index in range(first_index + 1, len(candidates)):
                distance = self.candidate_distance(
                    candidates[first_index],
                    candidates[second_index],
                )
                if distance > best_distance:
                    best_distance = distance
                    best_pair = (first_index, second_index)

        selected_indices = [best_pair[0], best_pair[1]]
        selected_index_set = set(selected_indices)
        selection_scores = {
            best_pair[0]: best_distance,
            best_pair[1]: best_distance,
        }

        while len(selected_indices) < selected_capture_count:
            best_index = None
            best_min_distance = -1.0
            for candidate_index, candidate in enumerate(candidates):
                if candidate_index in selected_index_set:
                    continue
                min_distance_to_selected = min(
                    self.candidate_distance(candidate, candidates[selected_index])
                    for selected_index in selected_indices
                )
                if min_distance_to_selected > best_min_distance:
                    best_index = candidate_index
                    best_min_distance = min_distance_to_selected

            if best_index is None:
                break
            selected_indices.append(best_index)
            selected_index_set.add(best_index)
            selection_scores[best_index] = best_min_distance

        selected_candidates = []
        for order, candidate_index in enumerate(selected_indices, start=1):
            candidate = candidates[candidate_index]
            candidate.record['selected_for_calibration'] = True
            candidate.record['selection_order'] = order
            candidate.record['selection_distance_score'] = selection_scores[candidate_index]
            selected_candidates.append(candidate)

        return selected_candidates

    def write_result_files(
        self,
        working_directory: str,
        response,
        captures: list[dict],
        selected_candidates: list[CaptureCandidate],
    ) -> None:
        result_path = Path(working_directory) / self.get_string('result_json_filename')
        result_data = {
            'configuration': 'eye_to_hand',
            'meaning': 'camera pose in robot base frame',
            'selection_method': 'greedy_farthest_point_robot_and_board_pose_diversity',
            'selected_capture_handles': [
                candidate.capture_handle
                for candidate in selected_candidates
            ],
            'selected_pose_indices': [
                candidate.index
                for candidate in selected_candidates
            ],
            'transform_ros_m': transform_to_dict(response.transform),
            'residuals': [
                {
                    'rotation_deg': float(residual.rotation),
                    'translation_m': float(residual.translation),
                }
                for residual in response.residuals
            ],
            'captures': captures,
            'zivid_message': response.message,
        }
        with open(result_path, 'w', encoding='utf-8') as file:
            json.dump(result_data, file, indent=2)
            file.write('\n')
        self.get_logger().info(f'Wrote raw result JSON: {result_path}')

        yaml_path = Path(working_directory) / self.get_string('legacy_tf_yaml_filename')
        transform = response.transform
        yaml_text = (
            '/**:\n'
            '  ros__parameters:\n'
            f'    frame_id: "{self.get_string("legacy_tf_parent_frame")}"\n'
            f'    child_frame_id: "{self.get_string("legacy_tf_child_frame")}"\n'
            '    translation:\n'
            f'      x: {float(transform.translation.x) * 1000.0:.9g}\n'
            f'      y: {float(transform.translation.y) * 1000.0:.9g}\n'
            f'      z: {float(transform.translation.z) * 1000.0:.9g}\n'
            '    rotation:\n'
            f'      x: {float(transform.rotation.x):.9g}\n'
            f'      y: {float(transform.rotation.y):.9g}\n'
            f'      z: {float(transform.rotation.z):.9g}\n'
            f'      w: {float(transform.rotation.w):.9g}\n'
        )
        with open(yaml_path, 'w', encoding='utf-8') as file:
            file.write(yaml_text)
        self.get_logger().info(f'Wrote legacy millimeter TF YAML: {yaml_path}')

    def run(self) -> int:
        working_directory = self.make_working_directory()
        min_successful_captures = self.get_int('min_successful_captures')
        max_successful_captures = self.get_int('max_successful_captures')
        selected_capture_count = self.get_int('selected_capture_count')

        if min_successful_captures < 2:
            self.get_logger().error('min_successful_captures must be at least 2')
            return 2
        if max_successful_captures < 0:
            self.get_logger().error('max_successful_captures must be 0 or greater')
            return 2
        if 0 < max_successful_captures < min_successful_captures:
            self.get_logger().error(
                'max_successful_captures must be 0 or at least min_successful_captures'
            )
            return 2
        if selected_capture_count < 0:
            self.get_logger().error('selected_capture_count must be 0 or greater')
            return 2
        if 0 < selected_capture_count < min_successful_captures:
            self.get_logger().error(
                'selected_capture_count must be 0 or at least min_successful_captures'
            )
            return 2

        joint_configurations = validate_joint_configurations(
            load_joint_configurations(self.get_string('joint_configurations_file'))
        )
        self.warn_if_duplicate_poses(joint_configurations)

        if len(joint_configurations) < min_successful_captures:
            self.get_logger().warn(
                f'The joint list has only {len(joint_configurations)} poses, but '
                f'min_successful_captures is {min_successful_captures}. '
                'Calibration can only run if enough poses detect the board.'
            )

        if not self.configure_zivid():
            return 1
        if not self.start_hand_eye_session(working_directory):
            return 1
        if not self.connect_robot():
            return 1

        if self.get_bool('enable_keyboard_skip'):
            self.keyboard_skip.start()
        else:
            self.keyboard_skip.enabled = False

        captures = []
        candidates: list[CaptureCandidate] = []
        accepted_robot_poses = []
        try:
            for index, joints_deg in enumerate(joint_configurations):
                self.keyboard_skip.clear()
                if self.keyboard_skip.finish_collection_requested():
                    self.get_logger().warn(
                        'Stopping pose iteration before the next move because ENTER was pressed.'
                    )
                    break

                self.get_logger().info(
                    f'--- Capture {index + 1}/{len(joint_configurations)} ---'
                )
                if self.get_bool('move_robot'):
                    move_status = self.move_to_joints(joints_deg)
                    if move_status == 'finish':
                        self.get_logger().warn(
                            f'Pose {index + 1} interrupted by ENTER; '
                            'stopping candidate collection.'
                        )
                        captures.append(
                            {
                                **self.operator_skip_record(index, joints_deg),
                                'skip_reason': 'operator_finish_collection',
                            }
                        )
                        self.recover_after_keyboard_skip()
                        break
                    if move_status == 'skip':
                        self.get_logger().warn(
                            f'Pose {index + 1} skipped by operator before capture.'
                        )
                        captures.append(self.operator_skip_record(index, joints_deg))
                        self.recover_after_keyboard_skip()
                        continue
                    if move_status != 'ok':
                        return 1
                else:
                    self.get_logger().warn(
                        'move_robot is false; capturing at the current robot pose'
                    )

                meca_pose = [float(value) for value in self.get_robot_pose()]
                robot_pose = make_pose_from_meca_pose_meters(meca_pose)
                self.get_logger().info(
                    'Robot pose for Zivid [m, quat xyzw]: '
                    f'x={robot_pose.position.x:.6f}, y={robot_pose.position.y:.6f}, '
                    f'z={robot_pose.position.z:.6f}, qx={robot_pose.orientation.x:.6f}, '
                    f'qy={robot_pose.orientation.y:.6f}, qz={robot_pose.orientation.z:.6f}, '
                    f'qw={robot_pose.orientation.w:.6f}'
                )

                if self.keyboard_skip.requested():
                    self.get_logger().warn(
                        f'Pose {index + 1} skipped by operator before capture.'
                    )
                    captures.append(
                        self.operator_skip_record(index, joints_deg, meca_pose, robot_pose)
                    )
                    self.recover_after_keyboard_skip()
                    continue

                if self.keyboard_skip.finish_collection_requested():
                    self.get_logger().warn(
                        f'Pose {index + 1} interrupted by ENTER before capture; '
                        'stopping candidate collection.'
                    )
                    captures.append(
                        {
                            **self.operator_skip_record(index, joints_deg, meca_pose, robot_pose),
                            'skip_reason': 'operator_finish_collection',
                        }
                    )
                    self.recover_after_keyboard_skip()
                    break

                duplicate_index = self.duplicate_robot_pose_index(
                    robot_pose,
                    accepted_robot_poses,
                )
                if duplicate_index is not None:
                    self.get_logger().warn(
                        'Skipping capture because this robot pose duplicates accepted '
                        f'capture {duplicate_index + 1}.'
                    )
                    captures.append(
                        {
                            'index': index,
                            'commanded_joints_deg': joints_deg,
                            'meca_pose_mm_deg': meca_pose,
                            'robot_pose_ros_m': pose_to_dict(robot_pose),
                            'success': False,
                            'candidate': False,
                            'accepted': False,
                            'selected_for_calibration': False,
                            'capture_handle': -1,
                            'skipped': True,
                            'skip_reason': 'duplicate_robot_pose',
                        }
                    )
                    continue

                response = self.capture(robot_pose)
                if response is None:
                    return 1

                detection = response.detection_result_calibration_board
                capture_record = {
                    'index': index,
                    'commanded_joints_deg': joints_deg,
                    'meca_pose_mm_deg': meca_pose,
                    'robot_pose_ros_m': pose_to_dict(robot_pose),
                    'success': bool(response.success),
                    'candidate': False,
                    'accepted': False,
                    'selected_for_calibration': False,
                    'capture_handle': int(response.capture_handle),
                    'message': response.message,
                    'detection_status': int(detection.status),
                    'detection_status_description': detection.status_description,
                }
                captures.append(capture_record)

                if self.keyboard_skip.finish_collection_requested():
                    capture_record['skipped'] = True
                    capture_record['skip_reason'] = 'operator_finish_collection'
                    self.get_logger().warn(
                        f'Pose {index + 1} interrupted by ENTER after capture; '
                        'capture will not be used.'
                    )
                    self.recover_after_keyboard_skip()
                    break

                if self.keyboard_skip.requested():
                    capture_record['skipped'] = True
                    capture_record['skip_reason'] = 'operator_keyboard_skip'
                    self.get_logger().warn(
                        f'Pose {index + 1} skipped by operator after capture; '
                        'capture will not be used.'
                    )
                    self.recover_after_keyboard_skip()
                    continue

                if not response.success:
                    capture_record['skipped'] = True
                    capture_record['skip_reason'] = 'capture_or_detection_failed'
                    self.get_logger().warn(
                        'Skipping this pose; Zivid capture/detection failed: '
                        f'{response.message}'
                    )
                    continue
                if detection.status != DetectionResultCalibrationBoard.STATUS_OK:
                    capture_record['skipped'] = True
                    capture_record['skip_reason'] = 'board_not_detected_or_low_quality'
                    self.get_logger().warn(
                        'Skipping this pose; board detection is not OK: '
                        f'{detection.status_description}'
                    )
                    continue

                capture_record['candidate'] = True
                capture_record['accepted'] = True
                capture_record['skipped'] = False
                candidates.append(
                    CaptureCandidate(
                        index=index,
                        capture_handle=int(response.capture_handle),
                        robot_pose=copy_pose(robot_pose),
                        board_pose=copy_pose(detection.pose),
                        board_centroid=copy_point(detection.centroid),
                        record=capture_record,
                    )
                )
                accepted_robot_poses.append(robot_pose)
                self.get_logger().info(
                    f'Accepted capture handle {response.capture_handle} '
                    f'({len(candidates)} valid candidates)'
                )

                if (
                    max_successful_captures > 0
                    and len(candidates) >= max_successful_captures
                ):
                    self.get_logger().info(
                        f'Reached max_successful_captures={max_successful_captures}; '
                        'stopping candidate collection.'
                    )
                    break

            self.get_logger().info(
                f'Processed {len(captures)} poses; accepted '
                f'{len(candidates)} valid board-detection candidates.'
            )

            if len(candidates) < min_successful_captures:
                self.get_logger().error(
                    f'Need at least {min_successful_captures} successful distinct '
                    f'calibration-board captures, got {len(candidates)}. '
                    'Add more joint poses where the board is visible.'
                )
                return 1

            selected_candidates = self.select_calibration_candidates(
                candidates,
                selected_capture_count,
            )
            selected_capture_handles = [
                candidate.capture_handle
                for candidate in selected_candidates
            ]
            self.get_logger().info(
                'Selected capture handles for calibration: '
                + ', '.join(str(handle) for handle in selected_capture_handles)
            )

            self.get_logger().info('Calling Zivid hand-eye calibration')
            response = self.calibrate(selected_capture_handles)
            if response is None:
                return 1
            if not response.success:
                self.get_logger().error(f'Calibration failed: {response.message}')
                return 1

            transform = response.transform
            self.get_logger().info(
                'Calibration transform, camera pose in meca base [m, quat xyzw]: '
                f'x={transform.translation.x:.6f}, y={transform.translation.y:.6f}, '
                f'z={transform.translation.z:.6f}, qx={transform.rotation.x:.6f}, '
                f'qy={transform.rotation.y:.6f}, qz={transform.rotation.z:.6f}, '
                f'qw={transform.rotation.w:.6f}'
            )
            for index, residual in enumerate(response.residuals):
                if index < len(selected_candidates):
                    selected_candidates[index].record['residual_rotation_deg'] = float(
                        residual.rotation
                    )
                    selected_candidates[index].record['residual_translation_m'] = float(
                        residual.translation
                    )
                self.get_logger().info(
                    f'Residual {index}: rotation={residual.rotation:.4f} deg, '
                    f'translation={residual.translation * 1000.0:.3f} mm'
                )

            self.write_result_files(
                working_directory,
                response,
                captures,
                selected_candidates,
            )
            return 0
        finally:
            self.keyboard_skip.stop()
            self.disconnect_robot()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutomatedEyeToHand()
    try:
        exit_code = node.run()
    except KeyboardInterrupt:
        node.get_logger().warn('Interrupted by user')
        exit_code = 130
    except Exception as error:
        node.get_logger().error(str(error))
        exit_code = 1
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
