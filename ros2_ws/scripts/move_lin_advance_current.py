#!/usr/bin/env python3
"""
move_lin_advance_current.py
---------------------------

Publish a relative native Meca500 MoveLin advance command to convert_to_meca_node.

The motion is computed INSIDE convert_to_meca_node from the robot's real current
TCP pose using GetPose(), then executed as:

    current TCP pose + distance_mm * axis_sign * current_tcp_axis[axis_column]

Topic:
    /meca_lin_advance

Message:
    std_msgs/Float64MultiArray
    [distance_mm, axis_column, axis_sign]

This avoids opening a second connection to the Meca500 and avoids using a stale
or differently-corrected T_base_target.npy for the native MoveLin step.
"""

import argparse
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class AdvancePublisher(Node):
    def __init__(self, topic: str):
        super().__init__('meca_movelin_advance_current_publisher')
        self.pub = self.create_publisher(Float64MultiArray, topic, 10)

    def publish_once(
        self,
        distance_mm: float,
        axis_column: int,
        axis_sign: float,
        discovery_wait_sec: float,
        command_id=None,
    ) -> bool:
        """
        Publish one relative advance command robustly.

        This command is RELATIVE, so duplicate delivery would execute the
        advance multiple times. Therefore we publish retries only with a stable
        command_id, and convert_to_meca_node ignores duplicated command IDs.

        Before publishing, wait until DDS has actually discovered at least one
        subscriber. A fixed sleep is not robust enough with FastDDS under load.
        """

        # Wait until at least one subscriber is discovered.
        # Keep a minimum 2 s deadline even if the CLI default is smaller.
        wait_deadline = time.time() + max(2.0, float(discovery_wait_sec))

        while self.pub.get_subscription_count() == 0:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() > wait_deadline:
                self.get_logger().warn(
                    f"{self.pub.topic_name}: no subscriber found after "
                    f"{max(2.0, float(discovery_wait_sec)):.1f} s — command NOT sent."
                )
                return False

        if command_id is None:
            command_id = int(time.time() * 1000)

        msg = Float64MultiArray()
        msg.data = [
            float(distance_mm),
            float(axis_column),
            float(axis_sign),
            float(command_id),
        ]

        self.get_logger().info(
            f"Subscriber discovered on {self.pub.topic_name}. "
            f"Publishing command_id={int(command_id)} with duplicate-safe retries."
        )

        # Publish retries for DDS reliability. They are safe because all retries
        # share the same command_id and are deduplicated by convert_to_meca_node.
        for _ in range(3):
            self.pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.1)

        return True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='MoveLin advance from the current TCP pose via convert_to_meca_node.'
    )
    parser.add_argument('--distance-mm', '--advance-mm', dest='distance_mm', type=float, default=5.0)
    parser.add_argument('--axis-column', type=int, default=2, choices=[0, 1, 2],
                        help='Current TCP axis column: 0=X, 1=Y, 2=Z. Default: 2 (+Z insertion axis with SetTrf(0,0,L)).')
    parser.add_argument('--axis-sign', type=float, default=1.0, choices=[-1.0, 1.0])
    parser.add_argument('--topic', default='/meca_lin_advance')
    parser.add_argument('--discovery-wait-sec', type=float, default=0.5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    print('Relative MoveLin advance from CURRENT TCP pose')
    print('----------------------------------------------')
    print(f'topic:       {args.topic}')
    print(f'distance:    {args.distance_mm:.3f} mm')
    print(f'axis_column: {args.axis_column}  (0=X, 1=Y, 2=Z current TCP frame)')
    print(f'axis_sign:   {args.axis_sign:+.0f}')
    print('')

    # Unique ID for this CLI call. All retry publishes share this ID so the
    # subscriber can safely discard duplicates.
    command_id = int(time.time() * 1000)
    print(f'command_id:  {command_id}')
    print('')

    rclpy.init(args=None)
    node = AdvancePublisher(args.topic)
    try:
        sent = node.publish_once(
            distance_mm=args.distance_mm,
            axis_column=args.axis_column,
            axis_sign=args.axis_sign,
            discovery_wait_sec=args.discovery_wait_sec,
            command_id=command_id,
        )
        if sent:
            print('Command published with duplicate-safe command_id retry.')
        else:
            print('Command NOT sent: no subscriber discovered.')
            sys.exit(2)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(sys.argv[1:])
