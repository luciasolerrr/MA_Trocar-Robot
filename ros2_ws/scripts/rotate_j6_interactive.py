#!/usr/bin/env python3
"""
rotate_j6_interactive.py
------------------------
Control continuo de J6 con flechas de teclado para alinear la aguja.

El panel se redibuja en su sitio (sin scroll) a ~25 Hz.
Los comandos al robot se envían con debounce de DEBOUNCE_SEC tras la última tecla,
para no saturar la cola de convert_to_meca_node con una ráfaga de MoveJoints.

Teclas:
    ↑ / ↓       Girar J6 ± paso
    ← / →       (alias de ↑ / ↓, por comodidad)
    + / =       Aumentar paso
    - / _       Reducir paso
    r           Resetear destino al ángulo actual del robot
    ENTER       Si hay cambio pendiente: enviar ahora.
                Si no hay cambio: confirmar y salir.
    q / Ctrl+C  Salir sin confirmar (el pipeline continúa igualmente).
"""

import os
import sys
import tty
import termios
import select
import time
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros de control
# ─────────────────────────────────────────────────────────────────────────────

STEP_INIT  =  5.0   # grados por tecla (valor inicial)
STEP_MIN   =  0.5
STEP_MAX   = 20.0
STEP_INC   =  0.5   # cambio al pulsar + / -

J6_MIN_DEG = -700.0
J6_MAX_DEG =  700.0

DEBOUNCE_SEC = 0.35  # s — espera tras última tecla antes de enviar al robot
IDLE_WIN_SEC =  0.3  # s — ventana temporal para detectar si J6 está en reposo
IDLE_THR_DEG =  0.2  # ° — variación máxima de J6 para considerarlo "en reposo"

# Geometría del panel ASCII
PANEL_W = 48   # ancho interior (entre ║ y ║, sin contar los propios ║)


# ─────────────────────────────────────────────────────────────────────────────
# Nodo ROS 2
# ─────────────────────────────────────────────────────────────────────────────

class J6Node(Node):
    """
    Suscribe /joint_states y publica en /joint_targets.
    Todo acceso a estado compartido está protegido por self._lk.
    """

    def __init__(self):
        super().__init__("rotate_j6_interactive")

        self._lk   = threading.Lock()
        self._j    = None                       # [j1…j6] en grados
        self._j6h  = []                         # historial (t_mono, j6_deg)

        self.sub = self.create_subscription(
            JointState, "/joint_states", self._cb_joints, 10
        )
        self.pub = self.create_publisher(
            Float64MultiArray, "/joint_targets", 10
        )

    def _cb_joints(self, msg: JointState):
        j = [float(np.degrees(v)) for v in msg.position]
        t = time.monotonic()
        with self._lk:
            self._j = j
            self._j6h.append((t, j[5]))
            self._j6h = [(ts, v) for ts, v in self._j6h if t - ts < 2.0]

    # ── Propiedades de sólo lectura (thread-safe) ─────────────────────────────

    @property
    def joints_deg(self):
        """Copia de los 6 ángulos articulares en grados, o None."""
        with self._lk:
            return list(self._j) if self._j is not None else None

    @property
    def j6_deg(self):
        """Ángulo J6 actual en grados, o None."""
        with self._lk:
            return float(self._j[5]) if self._j is not None else None

    @property
    def is_idle(self) -> bool:
        """True si J6 no se ha movido más de IDLE_THR_DEG en IDLE_WIN_SEC."""
        with self._lk:
            now = time.monotonic()
            rec = [v for ts, v in self._j6h if now - ts < IDLE_WIN_SEC]
        if len(rec) < 2:
            return True
        return (max(rec) - min(rec)) < IDLE_THR_DEG

    # ── Envío de comando ──────────────────────────────────────────────────────

    def send_j6(self, j6_target_deg: float):
        """
        Publica MoveJoints con J1–J5 del estado actual y J6 = j6_target_deg.
        Publica dos veces por fiabilidad DDS. El comando es absoluto
        (idempotente), así que duplicados son inocuos.
        """
        js = self.joints_deg
        if js is None:
            return
        cmd = list(js)
        cmd[5] = float(j6_target_deg)
        msg = Float64MultiArray()
        msg.data = [float(v) for v in cmd]
        self.pub.publish(msg)
        time.sleep(0.025)
        self.pub.publish(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Lectura de tecla sin bloqueo
# ─────────────────────────────────────────────────────────────────────────────

def read_key(fd: int, timeout: float = 0.04) -> bytes | None:
    """
    Lee una pulsación de teclado. Devuelve bytes o None si no hay nada.
    En modo raw, las teclas especiales (flechas) llegan como secuencias ESC.
    """
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    ch = os.read(fd, 1)
    if ch == b'\x1b':
        # Intentar leer el resto de la secuencia de escape (2 bytes más)
        r2, _, _ = select.select([sys.stdin], [], [], 0.05)
        if r2:
            return ch + os.read(fd, 2)
    return ch


# ─────────────────────────────────────────────────────────────────────────────
# Renderizado del panel
# ─────────────────────────────────────────────────────────────────────────────

def _box_row(text: str, w: int = PANEL_W) -> str:
    """Línea interior del cuadro con relleno hasta w caracteres."""
    return f"  \u2551 {text:<{w}} \u2551"   # ║ ... ║


def _build_panel(j6_actual, j6_target, step, idle, pending) -> list[str]:
    """Construye las líneas del panel. Devuelve una lista de strings."""
    W = PANEL_W

    # Campos con formato fijo para evitar saltos de ancho
    s_act = f"{j6_actual:+8.2f}\u00b0" if j6_actual is not None else "  N/A    "
    s_st  = "\u25cf reposo  " if idle    else "\u21bb moviendo"  # ● / ↻
    s_pd  = "\u25c6 PENDIENTE" if pending else "           "      # ◆

    lines = [
        f"  \u2554{'═' * (W + 2)}\u2557",                              # ╔═╗
        _box_row("  ROTACI\u00d3N J6  \u2014  alineaci\u00f3n de aguja"),  # título
        f"  \u2560{'═' * (W + 2)}\u2563",                              # ╠═╣
        _box_row(f"J6 actual:   {s_act}   {s_st}"),
        _box_row(f"J6 destino:  {j6_target:+8.2f}\u00b0   {s_pd}"),
        _box_row(f"Paso:        {step:5.1f}\u00b0"),
        f"  \u2560{'═' * (W + 2)}\u2563",                              # ╠═╣
        _box_row("\u2191 / \u2193     Girar J6 \u00b1 paso"),          # ↑↓ ±
        _box_row("+ / -     Aumentar / reducir el paso"),
        _box_row("r         Resetear destino al \u00e1ngulo actual"),   # ángulo
        _box_row("ENTER     Confirmar y continuar"),
        _box_row("q         Salir"),
        f"  \u255a{'═' * (W + 2)}\u255d",                              # ╚═╝
    ]
    return lines


# Número de líneas del panel (constante; se usa para mover el cursor hacia arriba).
_PANEL_LINES = len(_build_panel(0.0, 0.0, 5.0, True, False))


def render(j6_actual, j6_target, step, idle, pending, *, first: bool = False):
    """
    Dibuja el panel en la terminal.
    first=True → imprime desde la posición actual del cursor.
    first=False → sube _PANEL_LINES líneas y sobreescribe.
    """
    lines = _build_panel(j6_actual, j6_target, step, idle, pending)

    out = ""
    if not first:
        out += f"\033[{_PANEL_LINES}A"   # subir N líneas

    for ln in lines:
        # \033[2K limpia la línea completa antes de escribir (evita artefactos)
        out += f"\033[2K{ln}\r\n"

    sys.stdout.write(out)
    sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Sesión interactiva
# ─────────────────────────────────────────────────────────────────────────────

def run_session(node: J6Node) -> None:
    """
    Ejecuta la sesión interactiva de rotación J6.
    Bloquea hasta que el operador sale (ENTER sin cambio, o q).
    """

    # ── Esperar primer mensaje de joint_states (fuera de modo raw) ────────────
    print("\n  Conectando con /joint_states...")
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if node.j6_deg is not None:
            break
        rclpy.spin_once(node, timeout_sec=0.05)  # despacha callbacks DDS, incluido _cb_joints

    if node.j6_deg is None:
        print(
            "  [ERROR] No se recibe /joint_states.\n"
            "  ¿Está corriendo convert_to_meca_node?"
        )
        return

    j6_init = node.j6_deg
    print(f"  J6 actual: {j6_init:+.2f}°\n")
    print("  ⚠  El robot puede estar completando aún el MoveLin anterior.")
    input("     Pulsa ENTER cuando el robot esté completamente estático... ")
    print()

    # ── Estado interno del controlador de J6 ─────────────────────────────────
    j6_target = round(node.j6_deg or j6_init, 1)
    step      = STEP_INIT
    last_key  = 0.0
    pending   = False
    last_sent = j6_target
    confirmed = False   # se pone a True solo si el operador sale con ENTER

    # Primer dibujado del panel (en modo normal, antes de setraw)
    render(node.j6_deg, j6_target, step, node.is_idle, pending, first=True)

    # ── Entrar en modo raw ────────────────────────────────────────────────────
    fd  = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)

        while True:
            # ── Procesar mensajes ROS 2 pendientes (non-blocking) ─────────────
            rclpy.spin_once(node, timeout_sec=0.0)

            # ── Leer tecla (bloquea máx. DEBOUNCE_SEC/2 para no perder el envío)
            key = read_key(fd, timeout=0.04)

            if key is not None:
                if key in (b'\x1b[A', b'\x1b[C'):       # ↑ o →
                    j6_target = min(J6_MAX_DEG, round(j6_target + step, 1))
                    last_key  = time.monotonic()
                    pending   = True

                elif key in (b'\x1b[B', b'\x1b[D'):     # ↓ o ←
                    j6_target = max(J6_MIN_DEG, round(j6_target - step, 1))
                    last_key  = time.monotonic()
                    pending   = True

                elif key in (b'+', b'='):
                    step = min(STEP_MAX, round(step + STEP_INC, 1))

                elif key in (b'-', b'_'):
                    step = max(STEP_MIN, round(step - STEP_INC, 1))

                elif key == b'r':
                    j6n = node.j6_deg
                    if j6n is not None:
                        j6_target = round(j6n, 1)
                        last_sent = j6_target
                        pending   = False

                elif key in (b'\r', b'\n'):
                    if pending or abs(j6_target - last_sent) > 0.05:
                        # Forzar envío inmediato (saltar debounce)
                        last_key = 0.0
                    else:
                        # Sin cambio pendiente → confirmar y salir
                        confirmed = True
                        break

                elif key in (b'q', b'Q', b'\x03', b'\x04'):  # q, Ctrl+C/D
                    confirmed = False
                    break

            # ── Enviar al robot tras debounce ─────────────────────────────────
            if pending and (time.monotonic() - last_key) >= DEBOUNCE_SEC:
                if abs(j6_target - last_sent) > 0.05:
                    node.send_j6(j6_target)
                    last_sent = j6_target
                pending = False

            # ── Actualizar panel ──────────────────────────────────────────────
            render(node.j6_deg, j6_target, step, node.is_idle, pending)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        # Restaurar el cursor a modo normal justo después del panel
        sys.stdout.write("\r\n")
        sys.stdout.flush()

    if confirmed:
        print("  [OK] Alineación J6 confirmada. Continuando con inserción final...\n")
    else:
        print(
            "  [ABORT] Sesión J6 cancelada con 'q'. "
            "El MoveLin final NO se ejecutará.\n"
        )

    return confirmed


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys

    rclpy.init(args=None)
    node = J6Node()

    confirmed = False
    try:
        confirmed = run_session(node) or False
    except KeyboardInterrupt:
        # Ctrl+C fuera de raw mode: salir sin confirmar
        print("\n  [INTERRUPT] Sesión J6 interrumpida.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # Código de salida: 0 = confirmado (ENTER), 1 = cancelado (q / Ctrl+C)
    _sys.exit(0 if confirmed else 1)


if __name__ == "__main__":
    main()
