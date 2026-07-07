#!/usr/bin/env python3
"""
send_meca_move_pose_from_target.py
----------------------------------

Carga T_base_target (.npy o .json), lo convierte a pose cartesiana nativa
Meca500 [x_mm, y_mm, z_mm, alpha_deg, beta_deg, gamma_deg] y lo publica en:

    /meca_move_pose

El nodo convert_to_meca_node_native_movepose.py debe estar corriendo y conectado
al Meca500. No se abre una segunda conexión al robot.

Importante:
  - La traslación de T_base_target debe estar en mm.
  - La rotación se interpreta directamente como FRF/TRF nativo Meca500:
        +Z = eje de inserción / approach del TCP activo con SetTrf(0,0,95)
  - No aplica corrección URDF tcp_link/+X.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

try:
    from scipy.spatial.transform import Rotation
except Exception as e:
    Rotation = None
    _SCIPY_IMPORT_ERROR = e


def find_key_recursive(obj, key):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key_recursive(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_key_recursive(value, key)
            if found is not None:
                return found
    return None


def load_T_base_target(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    if path.suffix.lower() == ".npy":
        T = np.load(str(path))
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = find_key_recursive(data, "T_base_target")
        if raw is None:
            raise KeyError("No encuentro la clave T_base_target en el JSON")
        T = np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError("El target debe ser .npy o .json")

    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T_base_target debe ser 4x4, pero es {T.shape}")
    return T


class NativeMovePosePublisher(Node):
    def __init__(self, topic: str):
        super().__init__("send_meca_move_pose_from_target")
        self.pub = self.create_publisher(Float64MultiArray, topic, 10)

    def publish_once(self, goal, discovery_wait_sec: float):
        # Poll until the subscriber (convert_to_meca_node) is discovered by DDS,
        # instead of relying on a fixed timeout.  With VOLATILE durability, any
        # message published before the subscriber is matched is silently lost.
        # If the robot is already running (normal case), discovery takes <200 ms.
        # Hard deadline: max(discovery_wait_sec, 2.0) s to avoid hanging forever.
        deadline = time.time() + max(float(discovery_wait_sec), 2.0)
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.pub.get_subscription_count() > 0:
                break

        if self.pub.get_subscription_count() == 0:
            import sys
            print(
                f"ERROR: no subscriber found for /meca_move_pose after "
                f"{max(discovery_wait_sec, 2.0):.1f} s. "
                "Command NOT sent. Is convert_to_meca_node running?",
                file=sys.stderr,
            )
            sys.exit(1)

        msg = Float64MultiArray()
        msg.data = [float(v) for v in goal]

        # Publish a few times for DDS reliability.  MovePose is an absolute
        # command so duplicate delivery is safe (idempotent).
        for _ in range(3):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Publica T_base_target como MovePose nativo del Meca500."
    )
    parser.add_argument("--target", required=True, type=Path,
                        help="Ruta a T_base_target.npy o result_summary/prediction.json")
    parser.add_argument("--topic", default="/meca_move_pose")
    parser.add_argument("--discovery-wait-sec", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo imprime la pose; no publica en ROS.")
    parser.add_argument("--extra-clearance-mm", type=float, default=0.0,
                        help=(
                            "Retrocede la posición N mm hacia atrás a lo largo del eje de inserción "
                            "(-Z de T_base_target, alejándose del ojo). Usa 30 para T_stage. "
                            "Default 0: publica T_target tal cual."
                        ))
    parser.add_argument("--base-x-correction-mm", type=float, default=0.0,
                        help="Corrección opcional en X nativo Meca antes de publicar. Default 0.")
    parser.add_argument("--base-y-correction-mm", type=float, default=0.0,
                        help="Corrección opcional en Y nativo Meca antes de publicar. Default 0.")
    parser.add_argument("--base-z-correction-mm", type=float, default=0.0,
                        help="Corrección opcional en Z nativo Meca antes de publicar. Default 0.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if Rotation is None:
        raise RuntimeError(
            f"scipy no está disponible y se necesita para convertir matriz->Euler XYZ: {_SCIPY_IMPORT_ERROR}"
        )

    T = load_T_base_target(args.target)
    p_mm = T[:3, 3].astype(np.float64).copy()
    R = T[:3, :3].astype(np.float64)

    if args.extra_clearance_mm != 0.0:
        # Convention: column 2 (+Z) is the insertion axis, pointing towards the eye.
        # Moving away from the eye means moving along -Z of T_base_target.
        z_ins = R[:, 2].astype(np.float64)
        z_norm = np.linalg.norm(z_ins)
        if z_norm < 1e-9:
            raise ValueError("T_base_target column 2 has near-zero norm")
        z_ins = z_ins / z_norm
        p_mm -= float(args.extra_clearance_mm) * z_ins

    p_mm += np.array([
        args.base_x_correction_mm,
        args.base_y_correction_mm,
        args.base_z_correction_mm,
    ], dtype=np.float64)

    # Meca500: Euler intrínseco/mobile XYZ. En scipy, mayúsculas = intrínseco.
    a_deg, b_deg, c_deg = Rotation.from_matrix(R).as_euler("XYZ", degrees=True)

    goal = [p_mm[0], p_mm[1], p_mm[2], a_deg, b_deg, c_deg]

    print("Native Meca500 MovePose target")
    print("------------------------------")
    print(f"source: {args.target}")
    print(f"topic:  {args.topic}")
    print(f"extra clearance: {args.extra_clearance_mm:.3f} mm")
    print(
        "goal:   "
        f"x={goal[0]:.3f} mm, y={goal[1]:.3f} mm, z={goal[2]:.3f} mm, "
        f"a={goal[3]:.3f} deg, b={goal[4]:.3f} deg, c={goal[5]:.3f} deg"
    )

    if args.dry_run:
        print("dry-run: no se publica nada.")
        return

    # Append a unique command_id so convert_to_meca_node can deduplicate
    # the 3 copies sent for DDS reliability without executing MovePose 3×.
    command_id = float(time.time_ns())
    goal_with_id = goal + [command_id]

    rclpy.init(args=None)
    node = NativeMovePosePublisher(args.topic)
    try:
        node.publish_once(goal_with_id, args.discovery_wait_sec)
        print("Command published once.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
