import time
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from tf_transformations import quaternion_from_euler


# Example sequence of Cartesian poses
# Format: [x, y, z, alpha, beta, gamma]
# Units:
#   - Position: passed through unchanged (must match what the robot node expects)
#   - Orientation: degrees
SEQ_CART = [
    # [120.855, 272.787, 102.828, -138.387, 60.466, 46.014]
    #[190, 0, 308, -138.387, 60.466, 46.014]
    [120, 272, 102, 0, 90, 0],
    #[190, 0, 308, 0, 90, 0],
    # [0.0, 180.0, 230.0, -151.0, 0.0, 140.0],
]


class CartesianSeqPub(Node):
    def __init__(self):
        super().__init__('cartesian_seq_publisher')
        # This must match the subscriber in JointCommandNode
        self.pub = self.create_publisher(PoseStamped, 'pose_for_meca', 10)

    def send_seq(self, seq, gap_s=0.2):
        for x, y, z, a_deg, b_deg, c_deg in seq:
            # Convert Euler angles (degrees) to quaternion
            roll = math.radians(float(a_deg))
            pitch = math.radians(float(b_deg))
            yaw = math.radians(float(c_deg))
            qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'meca_base_link'  # optional, not used by the robot node

            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = float(z)

            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw

            self.pub.publish(msg)

            self.get_logger().info(
                f'Published PoseStamped to /pose_for_meca: '
                f'x={x}, y={y}, z={z}, '
                f'a={a_deg}°, b={b_deg}°, c={c_deg}°'
            )

            time.sleep(gap_s)


def main():
    rclpy.init()
    node = CartesianSeqPub()
    time.sleep(0.5)  # ensure subscriber is ready
    node.send_seq(SEQ_CART, gap_s=0.2)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
