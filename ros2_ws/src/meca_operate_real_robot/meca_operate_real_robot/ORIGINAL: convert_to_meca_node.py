#!/usr/bin/env python3

import threading
import queue
import math
import time

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, TransformStamped

from tf_transformations import euler_from_quaternion, quaternion_from_euler
from tf2_ros import TransformBroadcaster

from mecademicpy.robot import Robot


class JointCommandNode(Node):
    def __init__(self):
        super().__init__('joint_command_node_seq')

        # ---- Parameter ----
        self.declare_parameter('robot_ip', '192.168.0.100')
        self.declare_parameter('tf_parent_frame', 'meca_base_link')
        self.declare_parameter('tf_child_frame', 'flange')
        self.declare_parameter('tf_rate_hz', 60.0)

        robot_ip = self.get_parameter('robot_ip').value
        self.tf_parent_frame = self.get_parameter('tf_parent_frame').value
        self.tf_child_frame = self.get_parameter('tf_child_frame').value
        self.tf_rate_hz = float(self.get_parameter('tf_rate_hz').value)

        self.tf_period = 1.0 / self.tf_rate_hz if self.tf_rate_hz > 0.0 else 0.0167

        # Subscribers
        self.sub_joints = self.create_subscription(
            Float64MultiArray, 'joint_targets', self.cb_joints, 10
        )
        self.sub_pose = self.create_subscription(
            PoseStamped, 'pose_for_meca', self.cb_pose_stamped, 10
        )

        # Robot
        self.robot = Robot()
        self.robot.Connect(address=robot_ip)

        # Activation robust machen (dein Log zeigte 1013)
        try:
            self.robot.ActivateAndHome()
        except Exception as e:
            self.get_logger().warn(f'ActivateAndHome failed (continuing): {e}')

        try:
            self.robot.SetJointVel(1)
        except Exception as e:
            self.get_logger().warn(f'SetJointVel failed (continuing): {e}')

        # Monitoring-Rate setzen
        try:
            self.robot.SetMonitoringInterval(self.tf_period)
        except Exception as e:
            self.get_logger().warn(f'SetMonitoringInterval failed (continuing): {e}')

        # Lock nur für Command/Motion Calls
        self.robot_lock = threading.Lock()

        # Pose-Cache (x,y,z in Metern; Winkel in Grad)
        self.latest_pose_lock = threading.Lock()
        self.latest_pose = None  # (x,y,z,a,b,c)

        # Goal-Pose Cache (commanded target)
        self.latest_goal_pose_lock = threading.Lock()
        self.latest_goal_pose = None  # (x,y,z,a,b,c)

        # TF
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_timer = self.create_timer(self.tf_period, self.publish_tf_from_cache)

        # Worker Queue
        self.q = queue.Queue()
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

        # Pose monitor thread: liest Pose aus Monitoring-Snapshot
        self.pose_thread = threading.Thread(target=self.monitor_pose_loop, daemon=True)
        self.pose_thread.start()

        self.get_logger().info('Robot ready. TF is published from monitoring cache (updates during motion).')

    # ---------- Input Callbacks ----------

    def cb_joints(self, msg: Float64MultiArray):
        if len(msg.data) != 6:
            self.get_logger().error(f'Need 6 joint angles, got {len(msg.data)}.')
            return
        self.q.put(("joint", tuple(map(float, msg.data))))

    def cb_pose_stamped(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        a = math.degrees(roll)
        b = math.degrees(pitch)
        c = math.degrees(yaw)

        goal = (float(p.x), float(p.y), float(p.z), a, b, c)

        # Cache goal pose
        with self.latest_goal_pose_lock:
            self.latest_goal_pose = goal

        # Print/log goal pose (what the robot is going to)
        self.get_logger().info(
            f"Goal pose (cmd): x={goal[0]:.4f}, y={goal[1]:.4f}, z={goal[2]:.4f}, "
            f"a={goal[3]:.2f}°, b={goal[4]:.2f}°, c={goal[5]:.2f}°"
        )

        self.q.put(("cart", goal))

    # ---------- Monitoring (Robot -> Cache) ----------

    def monitor_pose_loop(self):
        """
        Holt Pose zyklisch aus dem letzten Monitoring-Snapshot (falls unterstützt),
        sonst Fallback auf blockierendes GetPose().
        """
        while rclpy.ok():
            try:
                try:
                    # bevorzugt: nicht-blockierend (Monitoring-Snapshot)
                    x, y, z, a, b, c = self.robot.GetPose(synchronous_update=False)
                except TypeError:
                    # Fallback für ältere mecademicpy-Versionen
                    x, y, z, a, b, c = self.robot.GetPose()

                with self.latest_pose_lock:
                    self.latest_pose = (float(x), float(y), float(z), float(a), float(b), float(c))

            except Exception as e:
                self.get_logger().error(f'Pose monitor failed: {e}')

            time.sleep(self.tf_period)

    # ---------- TF Publish (Cache -> /tf) ----------

    def publish_tf_from_cache(self):
        with self.latest_pose_lock:
            pose = self.latest_pose
        if pose is None:
            return

        x, y, z, a, b, c = pose
        qx, qy, qz, qw = quaternion_from_euler(
            math.radians(a), math.radians(b), math.radians(c)
        )

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.tf_parent_frame
        t.child_frame_id = self.tf_child_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z
        t.transform.rotation.x = qx
        t.transform.rotation.y = qy
        t.transform.rotation.z = qz
        t.transform.rotation.w = qw

        self.tf_broadcaster.sendTransform(t)

    # ---------- Worker (Motion Commands) ----------

    def run(self):
        while rclpy.ok():
            cmd_type, vals = self.q.get()
            try:
                with self.robot_lock:
                    if cmd_type == "joint":
                        self.robot.MoveJoints(*vals)

                    elif cmd_type == "cart":
                        x, y, z, a, b, c = vals
                        # Print/log the exact pose being executed (ordered, sequential)
                        self.get_logger().info(
                            f"Executing MovePose: x={x:.4f}, y={y:.4f}, z={z:.4f}, "
                            f"a={a:.2f}°, b={b:.2f}°, c={c:.2f}°"
                        )
                        self.robot.MovePose(*vals)

                    else:
                        self.get_logger().error(f'Unknown command type: {cmd_type}')
                        continue

                    # Sequenzierung beibehalten
                    self.robot.WaitIdle()

                    # Optional: log reached pose (from monitoring cache) after motion
                    with self.latest_pose_lock:
                        reached = self.latest_pose
                    if reached is not None:
                        rx, ry, rz, ra, rb, rc = reached
                        self.get_logger().info(
                            f"Reached pose (mon): x={rx:.4f}, y={ry:.4f}, z={rz:.4f}, "
                            f"a={ra:.2f}°, b={rb:.2f}°, c={rc:.2f}°"
                        )

            except Exception as e:
                self.get_logger().error(f'Motion command failed: {e}')
            finally:
                self.q.task_done()


def main(args=None):
    rclpy.init(args=args)
    node = JointCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

