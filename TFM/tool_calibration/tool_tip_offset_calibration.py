#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
from dataclasses import dataclass

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_matrix


def axis_from_name(axis_name: str) -> np.ndarray:
    axes = {
        'x': np.array([1.0, 0.0, 0.0], dtype=float),
        '+x': np.array([1.0, 0.0, 0.0], dtype=float),
        '-x': np.array([-1.0, 0.0, 0.0], dtype=float),
        'y': np.array([0.0, 1.0, 0.0], dtype=float),
        '+y': np.array([0.0, 1.0, 0.0], dtype=float),
        '-y': np.array([0.0, -1.0, 0.0], dtype=float),
        'z': np.array([0.0, 0.0, 1.0], dtype=float),
        '+z': np.array([0.0, 0.0, 1.0], dtype=float),
        '-z': np.array([0.0, 0.0, -1.0], dtype=float),
    }
    normalized = axis_name.strip().lower()
    if normalized not in axes:
        raise ValueError(
            f"Unsupported axis '{axis_name}'. Use x, y, z, +x, -x, +y, -y, +z, or -z."
        )
    return axes[normalized]


def format_vector(values: np.ndarray, precision: int = 3) -> str:
    return '[' + ', '.join(f'{float(value):.{precision}f}' for value in values) + ']'


def quat_multiply(lhs: tuple[float, float, float, float], rhs: tuple[float, float, float, float]):
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def quat_from_axis_angle(axis: tuple[float, float, float], angle_rad: float):
    ax, ay, az = axis
    half = 0.5 * angle_rad
    sin_half = math.sin(half)
    return (ax * sin_half, ay * sin_half, az * sin_half, math.cos(half))


def mecademic_intrinsic_xyz_to_rotation(
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
) -> np.ndarray:
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), math.radians(alpha_deg))
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(beta_deg))
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), math.radians(gamma_deg))
    q = quat_multiply(quat_multiply(qx, qy), qz)
    norm = math.sqrt(sum(component * component for component in q))
    if norm < 1e-12:
        q = (0.0, 0.0, 0.0, 1.0)
    else:
        q = tuple(component / norm for component in q)
    return quaternion_matrix(q)[:3, :3]


def sample_from_mecademic_pose(values: list[float]) -> PoseSample:
    if len(values) != 6:
        raise ValueError('Expected 6 values: x y z alpha beta gamma')
    x, y, z, alpha, beta, gamma = values
    return PoseSample(
        position_mm=np.array([x, y, z], dtype=float),
        rotation=mecademic_intrinsic_xyz_to_rotation(alpha, beta, gamma),
        stamp_s=0.0,
    )


@dataclass
class PoseSample:
    position_mm: np.ndarray
    rotation: np.ndarray
    stamp_s: float


class TipOffsetCalibrationNode(Node):
    def __init__(self, pose_topic: str):
        super().__init__('tool_tip_offset_calibration')
        self._lock = threading.Lock()
        self._latest_sample: PoseSample | None = None
        self._sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self._on_pose,
            10,
        )

    def _on_pose(self, msg: PoseStamped):
        q = msg.pose.orientation
        rotation = quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]
        position_mm = np.array(
            [
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(msg.pose.position.z),
            ],
            dtype=float,
        )
        stamp_s = float(msg.header.stamp.sec) + 1e-9 * float(msg.header.stamp.nanosec)
        with self._lock:
            self._latest_sample = PoseSample(position_mm, rotation, stamp_s)

    def latest_sample(self) -> PoseSample | None:
        with self._lock:
            if self._latest_sample is None:
                return None
            return PoseSample(
                self._latest_sample.position_mm.copy(),
                self._latest_sample.rotation.copy(),
                self._latest_sample.stamp_s,
            )


def solve_axis_length(samples: list[PoseSample], axis_local: np.ndarray) -> dict:
    rows = []
    rhs = []
    for sample in samples:
        axis_base = sample.rotation @ axis_local
        axis_norm = np.linalg.norm(axis_base)
        if axis_norm < 1e-12:
            raise RuntimeError('Encountered a degenerate transformed tool axis.')
        axis_base = axis_base / axis_norm
        block = np.zeros((3, 4), dtype=float)
        block[:, :3] = np.eye(3)
        block[:, 3] = -axis_base
        rows.append(block)
        rhs.append(sample.position_mm)

    matrix = np.vstack(rows)
    vector = np.concatenate(rhs)
    solution, _, rank, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
    pivot_mm = solution[:3]
    length_mm = float(solution[3])
    tips = np.array(
        [sample.position_mm + length_mm * (sample.rotation @ axis_local) for sample in samples],
        dtype=float,
    )
    residuals_mm = np.linalg.norm(tips - pivot_mm, axis=1)
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size > 0 and singular_values[-1] > 1e-12
        else math.inf
    )
    return {
        'mode': 'axis',
        'sample_count': len(samples),
        'rank': int(rank),
        'condition_number': condition,
        'pivot_point_base_mm': pivot_mm,
        'offset_along_axis_mm': length_mm,
        'residual_rms_mm': float(math.sqrt(float(np.mean(residuals_mm * residuals_mm)))),
        'residual_max_mm': float(np.max(residuals_mm)),
        'residuals_mm': residuals_mm,
    }


def solve_full_offset(samples: list[PoseSample], axis_local: np.ndarray) -> dict:
    rows = []
    rhs = []
    for sample in samples:
        block = np.zeros((3, 6), dtype=float)
        block[:, :3] = np.eye(3)
        block[:, 3:] = -sample.rotation
        rows.append(block)
        rhs.append(sample.position_mm)

    matrix = np.vstack(rows)
    vector = np.concatenate(rhs)
    solution, _, rank, singular_values = np.linalg.lstsq(matrix, vector, rcond=None)
    pivot_mm = solution[:3]
    offset_local_mm = solution[3:]
    tips = np.array(
        [sample.position_mm + sample.rotation @ offset_local_mm for sample in samples],
        dtype=float,
    )
    residuals_mm = np.linalg.norm(tips - pivot_mm, axis=1)
    length_along_axis_mm = float(offset_local_mm.dot(axis_local))
    lateral_offset_mm = float(
        np.linalg.norm(offset_local_mm - length_along_axis_mm * axis_local)
    )
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size > 0 and singular_values[-1] > 1e-12
        else math.inf
    )
    return {
        'mode': 'full',
        'sample_count': len(samples),
        'rank': int(rank),
        'condition_number': condition,
        'pivot_point_base_mm': pivot_mm,
        'offset_local_mm': offset_local_mm,
        'offset_along_axis_mm': length_along_axis_mm,
        'lateral_offset_mm': lateral_offset_mm,
        'residual_rms_mm': float(math.sqrt(float(np.mean(residuals_mm * residuals_mm)))),
        'residual_max_mm': float(np.max(residuals_mm)),
        'residuals_mm': residuals_mm,
    }


def serializable_result(result: dict) -> dict:
    converted = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            converted[key] = [float(component) for component in value.tolist()]
        elif isinstance(value, np.generic):
            converted[key] = float(value)
        else:
            converted[key] = value
    return converted


def print_result(result: dict, axis_name: str):
    print('')
    print(f"Result: {result['mode']} model")
    print(f"  samples: {result['sample_count']}")
    print(f"  rank: {result['rank']}")
    print(f"  condition_number: {result['condition_number']:.3f}")
    print(f"  pivot_point_base_mm: {format_vector(result['pivot_point_base_mm'])}")
    print(f"  offset_along_{axis_name}_mm: {result['offset_along_axis_mm']:.3f}")
    if result['mode'] == 'full':
        print(f"  offset_local_mm: {format_vector(result['offset_local_mm'])}")
        print(f"  lateral_offset_mm: {result['lateral_offset_mm']:.3f}")
    print(f"  residual_rms_mm: {result['residual_rms_mm']:.3f}")
    print(f"  residual_max_mm: {result['residual_max_mm']:.3f}")
    print(
        '  residuals_mm: '
        + '['
        + ', '.join(f'{float(value):.3f}' for value in result['residuals_mm'])
        + ']'
    )
    print(
        '  launch override: '
        f"tool_tip_offset_mm:={result['offset_along_axis_mm']:.3f}"
    )
    print(
        '  URDF needle_len suggestion: '
        f"{result['offset_along_axis_mm'] / 1000.0:.6f} m"
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Estimate needle/tool tip offset from pivot samples. Keep the physical '
            'tip fixed at one point, change robot orientation, and press Enter to '
            'capture each /meca_eef_pose sample.'
        )
    )
    parser.add_argument('--pose-topic', default='/meca_eef_pose')
    parser.add_argument('--axis', default='+z')
    parser.add_argument(
        '--input-mode',
        choices=['topic', 'manual'],
        default='topic',
        help=(
            'topic samples PoseStamped from --pose-topic; manual prompts for '
            'Mecademic x y z alpha beta gamma values.'
        ),
    )
    parser.add_argument(
        '--mode',
        choices=['axis', 'full', 'both'],
        default='both',
        help='axis solves only length along --axis; full solves a 3D local TCP offset.',
    )
    parser.add_argument('--min-samples', type=int, default=6)
    parser.add_argument(
        '--output',
        default='',
        help='Optional JSON output path for the calibration result.',
    )
    return parser.parse_args(argv)


def run_manual_input(args: argparse.Namespace, axis_local: np.ndarray):
    samples: list[PoseSample] = []
    print('')
    print('Tool tip offset calibration')
    print('  input_mode: manual')
    print(f'  axis: {args.axis}')
    print(f'  mode: {args.mode}')
    print('')
    print('Procedure:')
    print('  Do not start the ROS robot bridge for this mode.')
    print('  Move the robot with the Mecademic UI/controller.')
    print('  Keep the physical needle tip fixed at one point.')
    print('  Enter the Mecademic pose as: x y z alpha beta gamma')
    print('  Type s to solve, q to quit.')
    print(f'  Recommended minimum samples: {args.min_samples}')

    try:
        while True:
            command = input(
                f'\nSample {len(samples) + 1} '
                '[x y z alpha beta gamma, s=solve, q=quit]: '
            ).strip()
            if not command:
                continue
            lowered = command.lower()
            if lowered in ('q', 'quit', 'exit'):
                break
            if lowered in ('s', 'solve'):
                solve_and_print(samples, args, axis_local)
                continue

            try:
                values = [float(token) for token in command.replace(',', ' ').split()]
                sample = sample_from_mecademic_pose(values)
            except ValueError as error:
                print(f'Invalid sample: {error}')
                continue

            samples.append(sample)
            print(
                f'Captured #{len(samples)}: '
                f'position_mm={format_vector(sample.position_mm)}'
            )
            if len(samples) >= args.min_samples:
                print('Enough samples collected; type s to solve or capture more samples.')
    except KeyboardInterrupt:
        pass
    finally:
        if samples:
            print('\nFinal solve before exit:')
            solve_and_print(samples, args, axis_local)


def solve_and_print(samples: list[PoseSample], args: argparse.Namespace, axis_local: np.ndarray):
    if len(samples) < 2:
        print('Need at least 2 samples for the axis-length solve.')
        return None

    results = []
    if args.mode in ('axis', 'both'):
        axis_result = solve_axis_length(samples, axis_local)
        print_result(axis_result, args.axis)
        results.append(axis_result)

    if args.mode in ('full', 'both'):
        if len(samples) < 3:
            print('Need at least 3 samples for the full 3D offset solve.')
        else:
            full_result = solve_full_offset(samples, axis_local)
            print_result(full_result, args.axis)
            results.append(full_result)

    if args.output and results:
        payload = {
            'axis': args.axis,
            'pose_topic': args.pose_topic,
            'results': [serializable_result(result) for result in results],
        }
        with open(args.output, 'w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2)
            output_file.write('\n')
        print(f'\nWrote calibration result to {args.output}')

    return results


def main(argv=None):
    raw_argv = sys.argv if argv is None else [sys.argv[0], *argv]
    args = parse_args(remove_ros_args(args=raw_argv)[1:])
    axis_local = axis_from_name(args.axis)

    if args.input_mode == 'manual':
        run_manual_input(args, axis_local)
        return

    rclpy.init(args=raw_argv)
    node = TipOffsetCalibrationNode(args.pose_topic)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    samples: list[PoseSample] = []
    print('')
    print('Tool tip offset calibration')
    print(f'  pose_topic: {args.pose_topic}')
    print(f'  axis: {args.axis}')
    print(f'  mode: {args.mode}')
    print('')
    print('Procedure:')
    print('  Keep the physical needle tip fixed at one point.')
    print('  Move the flange through several different orientations.')
    print('  Press Enter for each sample, type s to solve, q to quit.')
    print(f'  Recommended minimum samples: {args.min_samples}')

    try:
        while rclpy.ok():
            command = input(f'\nSample {len(samples) + 1} [Enter=snapshot, s=solve, q=quit]: ')
            command = command.strip().lower()
            if command in ('q', 'quit', 'exit'):
                break
            if command in ('s', 'solve'):
                solve_and_print(samples, args, axis_local)
                continue

            sample = node.latest_sample()
            if sample is None:
                print(f'No pose received yet on {args.pose_topic}.')
                continue

            samples.append(sample)
            print(
                f'Captured #{len(samples)}: '
                f'position_mm={format_vector(sample.position_mm)}'
            )
            if len(samples) >= args.min_samples:
                print('Enough samples collected; type s to solve or capture more samples.')
    except KeyboardInterrupt:
        pass
    finally:
        if samples:
            print('\nFinal solve before exit:')
            solve_and_print(samples, args, axis_local)
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
