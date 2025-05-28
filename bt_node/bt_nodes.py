import time
from rclpy.node import Node
import py_trees
from geometry_msgs.msg import Twist, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool
from rclpy.duration import Duration
from rclpy.action import ActionClient
from bt_node.langchain_planner import get_plan
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus



class Wait(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="Wait"):
        super(Wait, self).__init__(name)
        self.node = node

    def update(self):
        return py_trees.common.Status.RUNNING

class MoveForward(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="MoveForward", speed=0.5, duration=3.0):
        super(MoveForward, self).__init__(name)
        self.node = node
        self.speed = speed
        self.duration = duration
        self._cmd_vel_pub = self.node.create_publisher(Twist, "/mobile_base_controller/cmd_vel_unstamped", 10)
        self._start_time = None
        self._timer = None

    def initialise(self):
        self._start_time = time.time()

        def publish_twist():
            twist = Twist()
            twist.linear.x = self.speed
            self._cmd_vel_pub.publish(twist)

        # Start a timer at 20Hz
        self._timer = self.node.create_timer(0.05, publish_twist)

    def update(self):
        if time.time() - self._start_time >= self.duration:
            print("MoveForward: Duration reached, stopping.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        # Stop publishing and stop the robot
        if self._timer is not None:
            self._timer.cancel()
        stop = Twist()
        self._cmd_vel_pub.publish(stop)


class LookAround(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="LookAround"):
        super().__init__(name)
        self.node = node
        self.publisher = self.node.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)
        self.executed = False
        self.start_time = None

    def initialise(self):
        self.executed = False
        self.start_time = time.time()

    def update(self):
        if not self.executed:
            traj = JointTrajectory()
            traj.joint_names = ['head_1_joint', 'head_2_joint']
            traj.points = [
                JointTrajectoryPoint(positions=[-1.2, 0.0], time_from_start=Duration(seconds=2.0).to_msg()),
                JointTrajectoryPoint(positions=[0.0, 0.0], time_from_start=Duration(seconds=4.0).to_msg()),
                JointTrajectoryPoint(positions=[1.2, 0.0], time_from_start=Duration(seconds=6.0).to_msg()),
                JointTrajectoryPoint(positions=[0.0, 0.0], time_from_start=Duration(seconds=8.0).to_msg())
            ]
            self.publisher.publish(traj)
            self.executed = True
            return py_trees.common.Status.RUNNING

        # Wait until 8 seconds have passed
        if time.time() - self.start_time >= 8.0:
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

class Condition_PersonDetected(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="Condition_PersonDetected"):
        super(Condition_PersonDetected, self).__init__(name)
        self.node = node
        self.person_detected = False
        self._person_sub = self.node.create_subscription(Bool, "/person_detected", self.person_callback, 10)

    def update(self):
        print("Person detection check...")
        if not self.person_detected:
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS
    
    def person_callback(self, msg: Bool):
        self.person_detected = msg.data

class NavigateTo(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="NavigateTo", x=2.0, y=2.0, frame_id="map"):
        super().__init__(name)
        self.node = node
        self.x = x  # (x, y, yaw)
        self.y = y
        self.frame_id = frame_id
        self._action_client = ActionClient(self.node, NavigateToPose, 'navigate_to_pose')
        self._goal_handle = None
        self._send_goal_future = None
        self._result_future = None
        self._status = py_trees.common.Status.RUNNING
        self.goal_sent = False

    def initialise(self):
        if self.goal_sent:
            return

        if not self._action_client.wait_for_server(timeout_sec=5.0):
            self.node.get_logger().error("NavigateToPose action server not available!")
            self._status = py_trees.common.Status.FAILURE
            return

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = self.frame_id
        goal_msg.pose.header.stamp = self.node.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.x
        goal_msg.pose.pose.position.y = self.y
        goal_msg.pose.pose.orientation.w = 1.0

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self.goal_sent = True
        self._status = py_trees.common.Status.RUNNING

    def update(self):
        if self._status != py_trees.common.Status.RUNNING:
            return self._status

        if self._send_goal_future is not None and self._goal_handle is None:
            if self._send_goal_future.done():
                goal_handle = self._send_goal_future.result()
                if not goal_handle.accepted:
                    self.node.get_logger().warn("Goal was rejected")
                    self._status = py_trees.common.Status.FAILURE
                else:
                    self._goal_handle = goal_handle
                    self._result_future = self._goal_handle.get_result_async()

        if self._result_future is not None:
            if self._result_future.done():
                result_msg = self._result_future.result()
                status_code = result_msg.status

                if status_code == GoalStatus.STATUS_SUCCEEDED:
                    self.node.get_logger().info("Navigation succeeded.")
                    self._status = py_trees.common.Status.SUCCESS
                else:
                    self.node.get_logger().warn(f"Navigation failed with status code: {status_code}")
                    self._status = py_trees.common.Status.FAILURE

        return self._status


    def terminate(self, new_status):
        self.goal_sent = False
        if new_status == py_trees.common.Status.INVALID and self._goal_handle:
            self._goal_handle.cancel_goal_async()
            self.node.get_logger().info("Navigation goal cancelled")
