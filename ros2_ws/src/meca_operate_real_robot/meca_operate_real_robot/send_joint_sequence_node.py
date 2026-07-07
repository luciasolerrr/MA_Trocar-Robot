# send_joint_sequence.py  (publisher)
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

SEQ = [
    [90, 0, -3.31875, 0, 65.29374, 49.61515],
    [90, 0.38664, -1.20025, 0, 62.78859, 49.61515],
    [90, 1.00395, 1.18725, 0, 59.78379, 49.61515],
    [90, 1.94774, 3.78297, 0, 56.24428, 49.61515],
    [90, 2.99338, 5.90428, 0, 53.07733, 49.61515],
    [0, 0, 0, 0, 0, 0],
    [90, 3.58161, 6.88191, 0, 51.51147, 49.61515],
]

#SEQ= [[0, 0, 0, 0, 0, 0],]

class JointSeqPub(Node):
    def __init__(self):
        super().__init__('joint_seq_publisher')
        self.pub = self.create_publisher(Float64MultiArray, 'joint_targets', 10)

    def send_seq(self, seq, gap_s=0.2):
        for joints in seq:
            msg = Float64MultiArray(data=[float(x) for x in joints])
            self.pub.publish(msg)
            self.get_logger().info(f'Published: {msg.data}')
            time.sleep(gap_s)  # spacing between messages

def main():
    rclpy.init()
    node = JointSeqPub()
    # small warmup so the subscriber is ready
    time.sleep(0.5)
    node.send_seq(SEQ, gap_s=0.2)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
