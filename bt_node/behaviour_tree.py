import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose
from turtlesim.srv import SetPen
import py_trees
import time
from bt_robot.langchain_planner import get_plan
import json

# move pose callback to check obstacle
# have a separate node for turning instead of moving forward

class MoveForward(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="MoveForward", speed=0.5, duration=2.0):
        super(MoveForward, self).__init__(name)
        self.node = node
        self.duration = duration
        self.speed = speed
        self.x = 1.0
        self.z = 0.0
        self._start_time = None

    def setup(self, timeout =10):
        self.publisher = self.node.create_publisher(Twist, '/cmd_vel', 10)
        return True

    def initialise(self):
        self._start_time = self.node.get_clock().now().seconds_nanoseconds()[0]

    def update(self):
        now = self.node.get_clock().now().seconds_nanoseconds()[0]
        elapsed = now - self._start_time

        self.node.get_logger().info(f"MoveForward: elapsed time = {elapsed} seconds")
        if elapsed < self.duration:
            twist = Twist()
            twist.linear.x = self.speed
            twist.angular.z = 0.0
            self.publisher.publish(twist)
            return py_trees.common.Status.RUNNING
        else:
            # stop the robot
            self.publisher.publish(Twist())
            self.node.get_logger().info("MoveForward completed")
            return py_trees.common.Status.SUCCESS
        
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
        self.publisher = node.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.start_time = None

    def initialise(self):
        self.start_time = time.time()

    def update(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 2.0  # Turn rate

        self.publisher.publish(msg)

        if time.time() - self.start_time > 1.6:  # ~180° at z=2.0
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.RUNNING
        
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


class BehaviorTreeRoot(Node):
    def __init__(self):
        super().__init__('bt_node')
        self.bt = self.create_behavior_tree()
        self.bt.setup(timeout=10)
        self.get_logger().info("Behavior Tree Initialized")
        self.timer = self.create_timer(1.0, self.tick_tree)

    def spin(self):
        rate = self.create_rate(10)  # 10 Hz
        while rclpy.ok():
            self.bt.tick()
            rclpy.spin_once(self)
            rate.sleep()
        
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

    def create_behavior_tree(self):
        root = py_trees.composites.Sequence(name="RootSequence", memory=True)
        root.add_children([MoveForward(self)])

        return py_trees.trees.BehaviourTree(root)

    def create_condition_node(self, condition_type):
        if condition_type == "Condition_CheckInBounds":
            return Condition_CheckInBounds(self)
        
        return Condition_CheckInBounds(self)
    
    def create_action_node(self, action_type, args=None):

        if action_type == "MoveForward":
            return MoveForward(self)
        elif action_type == "RotateLeft":
            return RotateLeft(self)
        elif action_type == "ChangePenColorCyclic":
            return ChangePenColorCyclic(self)
        elif action_type == "ChangePenColor" and args is not None and len(args) == 3:
            return ChangePenColor(self, args[0], args[1], args[2])
        
        return MoveForward(self)

    def bcreate_behavior_tree(self):
        plan = get_plan()
        print(plan)
        d = json.loads(plan)
        print(json.dumps(d, indent=4))

        root = self.create_behavior_tree_node(d)
        return py_trees.trees.BehaviourTree(root)    

    def bcreate_behavior_tree_node(self, d):
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
