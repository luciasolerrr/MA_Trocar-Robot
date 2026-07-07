#!/usr/bin/env python3

import argparse
import itertools
import json
import math
from pathlib import Path


DEFAULT_OFFSETS_DEG = [
    [-45.0, -25.0, 0.0, 25.0, 45.0],
    [-20.0, -10.0, 0.0, 10.0, 20.0],
    [-20.0, -10.0, 0.0, 10.0, 20.0],
    [-60.0, 0.0, 60.0],
    [-30.0, 0.0, 30.0],
    [-60.0, 0.0, 60.0],
]


def parse_float_list(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(',') if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError('Expected a comma-separated list of numbers')
    return values


def parse_joint_vector(text: str) -> list[float]:
    values = parse_float_list(text)
    if len(values) != 6:
        raise argparse.ArgumentTypeError(f'Expected 6 joint values, got {len(values)}')
    return values


def parse_joint_limits(text: str) -> list[tuple[float, float]]:
    pairs = []
    for item in text.split(';'):
        values = parse_float_list(item)
        if len(values) != 2:
            raise argparse.ArgumentTypeError(
                'Joint limits must use "min,max;min,max;..." with 6 pairs'
            )
        low, high = values
        if low > high:
            raise argparse.ArgumentTypeError(f'Invalid limit pair {item}: min > max')
        pairs.append((low, high))
    if len(pairs) != 6:
        raise argparse.ArgumentTypeError(f'Expected 6 joint-limit pairs, got {len(pairs)}')
    return pairs


def load_centers(path: str | None, cli_centers: list[list[float]]) -> list[list[float]]:
    if cli_centers:
        return cli_centers
    if not path:
        return [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    with open(Path(path).expanduser(), 'r', encoding='utf-8') as file:
        data = json.load(file)
    if isinstance(data, dict):
        data = data.get('joint_configurations_deg')
    if data is None:
        raise ValueError('Centers file must contain "joint_configurations_deg"')
    centers = [parse_joint_vector(','.join(str(value) for value in center)) for center in data]
    if not centers:
        raise ValueError('Centers file does not contain any center poses')
    return centers


def inside_limits(joints: list[float], limits: list[tuple[float, float]] | None) -> bool:
    if limits is None:
        return True
    return all(low <= value <= high for value, (low, high) in zip(joints, limits))


def dedupe_joint_poses(poses: list[list[float]]) -> list[list[float]]:
    seen = set()
    unique = []
    for pose in poses:
        key = tuple(round(value, 6) for value in pose)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pose)
    return unique


def rounded_pose(pose: list[float]) -> list[float]:
    return [round(float(value), 5) for value in pose]


def generate_grid(
    centers: list[list[float]],
    offsets_per_joint: list[list[float]],
    joint_limits: list[tuple[float, float]] | None,
) -> list[list[float]]:
    poses = []
    for center in centers:
        for offsets in itertools.product(*offsets_per_joint):
            pose = [center_value + offset for center_value, offset in zip(center, offsets)]
            if inside_limits(pose, joint_limits):
                poses.append(rounded_pose(pose))
    return dedupe_joint_poses(poses)


def normalized_joint_distance(
    first: list[float],
    second: list[float],
    scales: list[float],
) -> float:
    total = 0.0
    for first_value, second_value, scale in zip(first, second, scales):
        normalized = (first_value - second_value) / max(1e-9, scale)
        total += normalized * normalized
    return math.sqrt(total)


def farthest_point_subset(
    poses: list[list[float]],
    max_count: int,
    start_pose: list[float],
    scales: list[float],
) -> list[list[float]]:
    if max_count <= 0 or len(poses) <= max_count:
        return list(poses)

    start_index = min(
        range(len(poses)),
        key=lambda index: normalized_joint_distance(poses[index], start_pose, scales),
    )
    selected_indices = [start_index]
    selected_set = {start_index}

    while len(selected_indices) < max_count:
        best_index = None
        best_min_distance = -1.0
        for index, pose in enumerate(poses):
            if index in selected_set:
                continue
            min_distance = min(
                normalized_joint_distance(pose, poses[selected_index], scales)
                for selected_index in selected_indices
            )
            if min_distance > best_min_distance:
                best_index = index
                best_min_distance = min_distance
        if best_index is None:
            break
        selected_indices.append(best_index)
        selected_set.add(best_index)

    return [poses[index] for index in selected_indices]


def nearest_neighbor_order(
    poses: list[list[float]],
    start_pose: list[float],
    scales: list[float],
) -> list[list[float]]:
    remaining = list(poses)
    ordered = []
    current = start_pose
    while remaining:
        next_index = min(
            range(len(remaining)),
            key=lambda index: normalized_joint_distance(remaining[index], current, scales),
        )
        current = remaining.pop(next_index)
        ordered.append(current)
    return ordered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Generate systematic Mecademic joint configurations for Zivid hand-eye capture.'
    )
    parser.add_argument(
        '--output',
        default='hand_eye_joint_poses.generated.json',
        help='Output JSON path.',
    )
    parser.add_argument(
        '--center',
        action='append',
        type=parse_joint_vector,
        default=[],
        help='Center pose in degrees, e.g. "0,0,0,0,0,0". Can be repeated.',
    )
    parser.add_argument(
        '--centers-file',
        default='',
        help='JSON file with joint_configurations_deg used as center poses.',
    )
    for joint_index, default_offsets in enumerate(DEFAULT_OFFSETS_DEG, start=1):
        parser.add_argument(
            f'--j{joint_index}-offsets',
            type=parse_float_list,
            default=default_offsets,
            help='Comma-separated offsets in degrees.',
        )
    parser.add_argument(
        '--joint-limits',
        type=parse_joint_limits,
        default=None,
        help='Optional limits: "j1_min,j1_max;j2_min,j2_max;...;j6_min,j6_max".',
    )
    parser.add_argument(
        '--max-count',
        type=int,
        default=250,
        help='Select at most this many poses from the full grid. Use 0 for all.',
    )
    parser.add_argument(
        '--start',
        type=parse_joint_vector,
        default=None,
        help='Start pose used for ordering the output. Defaults to the first center pose.',
    )
    parser.add_argument(
        '--joint-scales',
        type=parse_joint_vector,
        default=[45.0, 20.0, 20.0, 60.0, 30.0, 60.0],
        help='Joint scales in degrees for diversity selection and ordering.',
    )
    parser.add_argument(
        '--no-order',
        action='store_true',
        help='Keep selected poses in diversity-selection order instead of nearest-neighbor order.',
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_count < 0:
        parser.error('--max-count must be 0 or greater')

    centers = load_centers(args.centers_file or None, args.center)
    offsets_per_joint = [
        args.j1_offsets,
        args.j2_offsets,
        args.j3_offsets,
        args.j4_offsets,
        args.j5_offsets,
        args.j6_offsets,
    ]

    grid = generate_grid(centers, offsets_per_joint, args.joint_limits)
    start_pose = args.start if args.start is not None else centers[0]
    selected = farthest_point_subset(grid, args.max_count, start_pose, args.joint_scales)
    if not args.no_order:
        selected = nearest_neighbor_order(selected, start_pose, args.joint_scales)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'joint_configurations_deg': [rounded_pose(pose) for pose in selected],
        'metadata': {
            'generator': 'generate_hand_eye_joint_grid',
            'centers_deg': centers,
            'offsets_deg': offsets_per_joint,
            'full_grid_count': len(grid),
            'output_count': len(selected),
            'max_count': args.max_count,
            'ordered_by_nearest_neighbor': not args.no_order,
        },
    }
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2)
        file.write('\n')

    print(f'Generated {len(selected)} poses from {len(grid)} grid candidates: {output_path}')


if __name__ == '__main__':
    main()
