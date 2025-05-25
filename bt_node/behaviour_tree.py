from rclpy.duration import Duration
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import SetPen
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import py_trees
import time
from bt_node.langchain_planner import get_plan
import json

# move pose callback to check obstacle
# have a separate node for turning instead of moving forward

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

class Condition_CheckInBounds(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="Condition_CheckInBounds"):
        super(Condition_CheckInBounds, self).__init__(name)
        self.obstacle_detected = False
        self.pose = None
        self.node = node
        self._pose_sub = self.node.create_subscription(Pose, "/turtle1/pose", self.pose_callback, 10)

    def update(self):
        print("Checking for obstacles...")
        if self.pose is None:
            return py_trees.common.Status.RUNNING
        if self.pose.x >= 9.0 or self.pose.x <= 2.0 or self.pose.y >= 9.0 or self.pose.y <= 2.0:
            return py_trees.common.Status.FAILURE
        
        return py_trees.common.Status.SUCCESS
    
    def pose_callback(self, pose: Pose):
        self.pose = pose

class TurnAround(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="TurnAround"):
        super().__init__(name)
        self.node = node
        self.angular_speed = 1.0  # radians per second
        self.duration = 3.14 / self.angular_speed
        self._start_time = None
        self._cmd_vel_pub = self.node.create_publisher(Twist, "/mobile_base_controller/cmd_vel_unstamped", 10)
        self._timer = None

    def initialise(self):
        self._start_time = time.time()

        def publish_twist():
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = self.angular_speed
            self._cmd_vel_pub.publish(twist)

        self._timer = self.node.create_timer(0.05, publish_twist)

    def update(self):
        if time.time() - self._start_time >= self.duration:
            print("TurnAround: Duration reached, stopping.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._timer is not None:
            self._timer.cancel()
        stop = Twist()
        self._cmd_vel_pub.publish(stop)
        
class RotateLeft(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="RotateLeft"):
        super().__init__(name)
        self.node = node
        self.publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()

    def update(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 1.0  # Turn rate

        self.publisher.publish(msg)

        if time.time() - self.start_time > 0.5:  # ~90° at z=2.0
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING      
                
class RotateRight(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="RotateRight"):
        super().__init__(name)
        self.node = node
        self.publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()

    def update(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = -1.0  # Turn rate

        self.publisher.publish(msg)

        if time.time() - self.start_time > 0.5:  # ~90° at z=-2.0
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING

class ChangePenColorCyclic(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, width=3, off=0):
        super().__init__(name="CyclePenColor")
        self.ros_node = node
        self.cli = self.ros_node.create_client(SetPen, '/turtle1/set_pen')

        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # R, G, B
        self.index = 0  # Current color index

        self.width = width
        self.off = off
        self.future = None
        self.done = False
        self.status_result = py_trees.common.Status.RUNNING

    def initialise(self):
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.ros_node.get_logger().error("Service /turtle1/set_pen not available")
            self.status_result = py_trees.common.Status.FAILURE
            self.done = True
            return

        r, g, b = self.colors[self.index]
        self.index = (self.index + 1) % len(self.colors)  # Advance to next color

        request = SetPen.Request()
        request.r = r
        request.g = g
        request.b = b
        request.width = self.width
        request.off = self.off

        self.ros_node.get_logger().info(f"Requesting pen color: R={r}, G={g}, B={b}")
        self.future = self.cli.call_async(request)
        self.done = False
        self.status_result = py_trees.common.Status.RUNNING

    def update(self):
        if self.done:
            return self.status_result

        if self.future.done():
            try:
                self.future.result()  # Check for exception
                self.status_result = py_trees.common.Status.SUCCESS
            except Exception as e:
                self.ros_node.get_logger().error(f"Service call failed: {e}")
                self.status_result = py_trees.common.Status.FAILURE
            self.done = True

        return self.status_result

class ChangePenColor(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node,  r, g, b, width=3, off=0):
        super().__init__(name="CyclePenColor")
        self.ros_node = node
        self.cli = self.ros_node.create_client(SetPen, '/turtle1/set_pen')
        self.colors = (r, g, b)

        self.width = width
        self.off = off
        self.future = None
        self.done = False
        self.status_result = py_trees.common.Status.RUNNING

    def initialise(self):
        if not self.cli.wait_for_service(timeout_sec=2.0):
            self.ros_node.get_logger().error("Service /turtle1/set_pen not available")
            self.status_result = py_trees.common.Status.FAILURE
            self.done = True
            return

        request = SetPen.Request()
        request.r, request.g, request.b = self.colors
        request.width = self.width
        request.off = self.off

        self.ros_node.get_logger().info(f"Requesting pen color: R={self.colors[0]}, G={self.colors[1]}, B={self.colors[2]}")
        self.future = self.cli.call_async(request)
        self.done = False
        self.status_result = py_trees.common.Status.RUNNING

    def update(self):
        if self.done:
            return self.status_result

        if self.future.done():
            try:
                self.future.result()  # Check for exception
                self.status_result = py_trees.common.Status.SUCCESS
            except Exception as e:
                self.ros_node.get_logger().error(f"Service call failed: {e}")
                self.status_result = py_trees.common.Status.FAILURE
            self.done = True

        return self.status_result

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
class BehaviorTreeRoot(Node):
    def __init__(self):
        super().__init__('bt_node')
        self.bt = self.create_behavior_tree()
        self.get_logger().info("Behavior Tree Initialized")
        self.timer = self.create_timer(1.0, self.tick_tree)
        
    # def create_behavior_tree(self):
    #     root = py_trees.composites.Selector(name="StayInBoundsOrTurn", memory=True)

    #     move_seq = py_trees.composites.Sequence(name="MoveIfInBounds", memory=True)
    #     move_seq.add_children([
    #         Condition_CheckInBounds(self),
    #         ChangePenColor(self, 0, 255, 0),
    #         MoveForward(self)
    #     ])

    #     turn_seq = py_trees.composites.Sequence(name="HandleOutOfBounds", memory=True)
    #     turn_seq.add_children([
    #         RotateLeft(self),
    #         ChangePenColor(self, 255, 0, 0),
    #         MoveForward(self)
    #     ])

        # root.add_children([move_seq, turn_seq])
        # return py_trees.trees.BehaviourTree(root)

    def bcreate_behavior_tree(self):
        root = py_trees.composites.Sequence(name="RootSequence", memory=True)
        root.add_children([MoveForward(self), TurnAround(self)])

        return py_trees.trees.BehaviourTree(root)

    def create_condition_node(self, condition_type):
        if condition_type == "Condition_CheckInBounds":
            return Condition_CheckInBounds(self)
        
        return Condition_CheckInBounds(self)
    
    def create_action_node(self, action_type, args=None):

        if action_type == "TurnAround":
            return TurnAround(self)
        elif action_type == "MoveForward":
            return MoveForward(self)
        elif action_type == "RotateLeft":
            return RotateLeft(self)
        elif action_type == "ChangePenColorCyclic":
            return ChangePenColorCyclic(self)
        elif action_type == "ChangePenColor" and args is not None and len(args) == 3:
            return ChangePenColor(self, args[0], args[1], args[2])
        
        return MoveForward(self)

    def create_behavior_tree(self):
        plan = get_plan()
        print(plan)
        d = json.loads(plan)
        print(json.dumps(d, indent=4))

        root = self.create_behavior_tree_node(d)
        return py_trees.trees.BehaviourTree(root)    

    def create_behavior_tree_node(self, d):
        root = None
        if not isinstance(d, dict):
            return None

        if "type" not in d:
            return None

        if d["type"].startswith("Condition_"):
            root = py_trees.composites.Selector(name="Selector", memory=True)
            true_child = py_trees.composites.Sequence(name="True Sequence", memory=True)
            false_child = py_trees.composites.Sequence(name="False Sequence", memory=True)

            node = self.create_condition_node(d["type"])
            true_child.add_child(node)

            for child in d.get("true", []):
                print(child)
                node = self.create_behavior_tree_node(child)
                if node is not None:
                    true_child.add_child(node)

            for child in d.get("false", []):
                print(child)
                node = self.create_behavior_tree_node(child)
                if node is not None:
                    false_child.add_child(node)

            root.add_children([true_child, false_child])
            return root
        
        elif d["type"] == "Sequence":
            root = py_trees.composites.Sequence(name="Sequence", memory=True)
            for child in d.get("true", []):
                node = self.create_behavior_tree_node(child)
                if node is not None:
                    root.add_child(node)
            return root

        args = d.get("args", None)
        root = self.create_action_node(d["type"], args)
        return root

    def tick_tree(self):
        self.bt.tick()

def main(args=None):
    rclpy.init(args=args)
    node = BehaviorTreeRoot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
