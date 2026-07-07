#!/usr/bin/env python3
"""
point_camera_to_meca.py

Transforma un punto 3D en el frame de la cámara (zivid_optical_frame)
al frame base del robot (meca_base_link) usando la calibración eye-to-hand,
construye la matriz T_base_tcp 4×4, la guarda como .npy y llama a
plan_moveit_then_send_to_meca.py.

Uso rápido:
    python3 point_camera_to_meca.py \
        --x 37.68292236328125 \
        --y -40.634422302246094 \
        --z 478.0090637207031

Para enviar al robot:
    python3 point_camera_to_meca.py \
        --x 37.68 --y -40.63 --z 478.01 \
        --send-to-meca

Calibración embebida (tf2_zivid_robotBase.generated.yaml):
    frame_id      : meca_base_link  (parent)
    child_frame_id: zivid_optical_frame
    translation   : x=-502.486969  y=229.313126  z=251.276108  [mm]
    quaternion    : x=0.583542057  y=-0.524915652  z=0.427242188  w=-0.448783168
"""

import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


# ──────────────────────────────────────────────────────────
# Calibración eye-to-hand (valores del YAML, mm + quat xyzw)
# ──────────────────────────────────────────────────────────
CALIB_TRANSLATION_MM = np.array([-502.486969, 229.313126, 251.276108])

CALIB_QUAT_XYZW = np.array([
     0.583542057,   # x
    -0.524915652,   # y
     0.427242188,   # z
    -0.448783168,   # w
])


# ──────────────────────────────────────────────────────────
# Utilidades de geometría
# ──────────────────────────────────────────────────────────

def quat_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """Quaternion [x, y, z, w] → matriz de rotación 3×3."""
    x, y, z, w = q_xyzw / np.linalg.norm(q_xyzw)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w)],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def build_T_base_camera() -> np.ndarray:
    """
    Construye T_base_camera (4×4, traslación en mm) a partir de la calibración.
    T_base_camera expresa la pose de zivid_optical_frame en meca_base_link.
    """
    R = quat_to_rot(CALIB_QUAT_XYZW)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = CALIB_TRANSLATION_MM
    return T


def point_camera_to_base(point_mm: np.ndarray) -> np.ndarray:
    """
    Aplica la transformación:
        P_base = R_base_cam * P_cam + t_base_cam
    Todo en mm.
    """
    T = build_T_base_camera()
    p_hom = np.append(point_mm, 1.0)
    return (T @ p_hom)[:3]


def default_approach_rotation() -> np.ndarray:
    """
    Orientación de aproximación por defecto:
    - El eje Z del TCP apunta hacia abajo (-Z del base) → típico para
      un robot montado sobre la mesa apuntando al objeto.
    - El eje X apunta hacia el frente del robot (+X del base).
    Cambia esto si tu setup es diferente (ver --qx/--qy/--qz/--qw).
    """
    z_tcp = np.array([0.0, 0.0, -1.0])   # TCP Z apunta "hacia abajo"
    x_tcp = np.array([1.0, 0.0,  0.0])   # TCP X apunta "al frente"
    y_tcp = np.cross(z_tcp, x_tcp)
    y_tcp /= np.linalg.norm(y_tcp)
    x_tcp = np.cross(y_tcp, z_tcp)
    x_tcp /= np.linalg.norm(x_tcp)
    # Columnas = ejes del frame TCP expresados en el base
    return np.column_stack([x_tcp, y_tcp, z_tcp])


def rotation_from_quat(qx, qy, qz, qw) -> np.ndarray:
    return quat_to_rot(np.array([qx, qy, qz, qw]))


def build_T_base_tcp(position_mm: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Construye la matriz T_base_tcp (4×4, traslación en mm)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position_mm
    return T


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )

    # Punto de entrada en frame cámara
    parser.add_argument("--x", type=float, required=True,
                        help="Coordenada X en zivid_optical_frame [mm]")
    parser.add_argument("--y", type=float, required=True,
                        help="Coordenada Y en zivid_optical_frame [mm]")
    parser.add_argument("--z", type=float, required=True,
                        help="Coordenada Z en zivid_optical_frame [mm]")

    # Orientación del TCP (opcional; si no se pasa se usa la por defecto)
    parser.add_argument("--qx", type=float, default=None,
                        help="Quaternion qx de la orientación del TCP en base frame")
    parser.add_argument("--qy", type=float, default=None,
                        help="Quaternion qy de la orientación del TCP en base frame")
    parser.add_argument("--qz", type=float, default=None,
                        help="Quaternion qz de la orientación del TCP en base frame")
    parser.add_argument("--qw", type=float, default=None,
                        help="Quaternion qw de la orientación del TCP en base frame")

    # Salida y planner
    parser.add_argument("--output-matrix", type=Path,
                        default=Path("/tmp/T_base_approach.npy"),
                        help="Ruta donde guardar la matriz .npy")
    parser.add_argument("--planner-script", type=Path,
                        default=Path(__file__).parent / "plan_moveit_then_send_to_meca.py",
                        help="Ruta al script plan_moveit_then_send_to_meca.py")

    # Parámetros del planner (se reenvían tal cual)
    parser.add_argument("--send-to-meca", action="store_true",
                        help="Enviar trayectoria al robot (pasa --send-to-meca al planner)")
    parser.add_argument("--group", default="meca_arm")
    parser.add_argument("--base-frame", default="meca_base_link")
    parser.add_argument("--target-link", default="tcp_link")
    parser.add_argument("--pos-tol", type=float, default=0.002,
                        help="Tolerancia de posición [m]")
    parser.add_argument("--orient-tol", type=float, default=0.08,
                        help="Tolerancia de orientación X/Y [rad]; Z siempre libre (π)")
    parser.add_argument("--velocity-scale", type=float, default=0.10)
    parser.add_argument("--acceleration-scale", type=float, default=0.10)
    parser.add_argument("--publish-gap", type=float, default=0.5)
    parser.add_argument("--include-first-point", action="store_true")

    args = parser.parse_args()

    # ── 1. Punto en frame cámara ──────────────────────────────────────
    p_cam = np.array([args.x, args.y, args.z])
    print(f"\n{'─'*60}")
    print(f"[1] Punto en zivid_optical_frame:  {p_cam} mm")

    # ── 2. Transformar al frame base ──────────────────────────────────
    p_base = point_camera_to_base(p_cam)
    print(f"[2] Punto en meca_base_link:       {np.round(p_base, 3)} mm")
    print(f"    (en metros):                   {np.round(p_base / 1000.0, 6)} m")

    # ── 3. Orientación del TCP ────────────────────────────────────────
    custom_quat = [args.qx, args.qy, args.qz, args.qw]
    if all(v is not None for v in custom_quat):
        R = rotation_from_quat(*custom_quat)
        print(f"[3] Orientación: quaternion personalizado {custom_quat}")
    else:
        R = default_approach_rotation()
        print(f"[3] Orientación: aproximación por defecto (TCP-Z apunta -Z base)")
        print(f"    Para cambiar, usa --qx --qy --qz --qw")

    # ── 4. Matriz T_base_tcp ──────────────────────────────────────────
    T = build_T_base_tcp(p_base, R)
    print(f"\n[4] T_base_tcp (4×4, traslación en mm):")
    print(np.round(T, 6))

    # Verificación rápida: la traslación debe tener sentido para el Meca500
    # (workspace típico: radio ~500 mm, altura hasta ~700 mm)
    dist_mm = np.linalg.norm(p_base)
    print(f"\n    Distancia al origen del robot: {dist_mm:.1f} mm")
    if dist_mm > 600:
        print("    ⚠️  ADVERTENCIA: punto fuera del workspace típico del Meca500 (~500 mm).")
    elif dist_mm < 50:
        print("    ⚠️  ADVERTENCIA: punto muy cerca del origen del robot.")
    else:
        print("    ✓  Distancia dentro del workspace típico.")

    # ── 5. Guardar .npy ───────────────────────────────────────────────
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(args.output_matrix), T)
    print(f"\n[5] Matriz guardada en: {args.output_matrix}")

    # ── 6. Llamar al planner ──────────────────────────────────────────
    if not args.planner_script.exists():
        print(f"\n[6] ⚠️  No se encontró el planner en: {args.planner_script}")
        print(f"    Ajusta --planner-script o copia plan_moveit_then_send_to_meca.py al lado.")
        print(f"\n    Puedes lanzarlo manualmente con:")
        _print_manual_cmd(args)
        sys.exit(0)

    cmd = [
        sys.executable, str(args.planner_script),
        "--matrix", str(args.output_matrix),
        "--units", "mm",
        "--group", args.group,
        "--base-frame", args.base_frame,
        "--target-link", args.target_link,
        "--pos-tol", str(args.pos_tol),
        "--orient-tol", str(args.orient_tol),
        "--velocity-scale", str(args.velocity_scale),
        "--acceleration-scale", str(args.acceleration_scale),
        "--publish-gap", str(args.publish_gap),
    ]
    if args.send_to_meca:
        cmd.append("--send-to-meca")
    if args.include_first_point:
        cmd.append("--include-first-point")

    print(f"\n[6] Lanzando planner:\n    {' '.join(str(c) for c in cmd)}\n")
    print(f"{'─'*60}\n")

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def _print_manual_cmd(args):
    print(
        f"    python3 plan_moveit_then_send_to_meca.py \\\n"
        f"        --matrix {args.output_matrix} \\\n"
        f"        --units mm \\\n"
        f"        --group {args.group} \\\n"
        f"        --base-frame {args.base_frame} \\\n"
        f"        --target-link {args.target_link} \\\n"
        f"        --pos-tol {args.pos_tol} \\\n"
        f"        --orient-tol {args.orient_tol} \\\n"
        f"        --velocity-scale {args.velocity_scale} \\\n"
        f"        --acceleration-scale {args.acceleration_scale}"
    )


if __name__ == "__main__":
    main()
