import time
from rclpy.node import Node
import py_trees
from geometry_msgs.msg import Twist, PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Duration
from visualization_msgs.msg import Marker
from rclpy.action import ActionClient
from bt_node.langchain_planner import get_plan
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus
import textwrap
from bt_node.config import Config

# MAIN NODES

#navigation
class NavigateTo(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="NavigateTo", room_id=Config.Room.HALL, frame_id="map"):
        super().__init__(name)
        self.node = node
        self.room_id = room_id
        room_goal = Config.rooms.get(room_id.value, (0.0, 0.0))
        self.x = room_goal[0]
        self.y = room_goal[1]
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
                    Config.current_room = self.room_id
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
#exploration
class ExploreFindPerson(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, person=Config.current_target, name="FindPerson"):
        super().__init__(name)
        self.node = node
        self.person = person
        self.subtree = self._build_subtree()

    def _build_subtree(self):
        root = py_trees.composites.Selector(name="FindPersonSelector", memory=True)

        for room in Config.Room:
            if room != Config.current_room:
                node = py_trees.composites.Sequence(name="ExploreSequence", memory=True)
                navigate = NavigateTo(self.node, room_id=room)
                node.add_child(navigate)
                node.add_child(Condition_PersonFound(self.node))
                node.add_child(Condition_FaceRecognized(self.node, person=self.person))
                root.add_child(node)
        
        return root
    
    def initialise(self):
        self.subtree.tick_once()

    def update(self):
        self.subtree.tick_once()
        status = self.subtree.status
        return status

    def terminate(self, new_status):
        self.subtree.stop(new_status)


# person & face recognition
class Condition_FaceRecognized(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, person=Config.current_target, name="Condition_FacialRecognition"):
        super().__init__(name)
        self.node = node
        self.person = person
        self.status = py_trees.common.Status.RUNNING

    def initialise(self):
        self.node.get_logger().info(f"Initialising Condition_FacialRecognition for {self.person}")

        # Knowing the target was in the current room, we check for a person here
        if Config.current_room in Config.people_rooms:
            # The person in the current room is the one we are looking for
            if Config.people_rooms[Config.current_room] == self.person:
                self.node.get_logger().info(f"Person {self.person} found in current room {Config.current_room}")
                Config.people_room_memory[self.person] = Config.current_room
                self.status = py_trees.common.Status.SUCCESS
            # There is another person in the current room, erase memory for this person
            # We could update the memory for the person we found, but we don't know who they are with the current tech
            else:
                self.node.get_logger().info(f"Person {self.person} not found in current room {Config.current_room}")
                if self.person in Config.people_room_memory:
                    del Config.people_room_memory[self.person]
                self.status = py_trees.common.Status.FAILURE
        # Out target has moved to another room, we need to look for them, erase memory 
        else:
            if self.person in Config.people_room_memory:
                del Config.people_room_memory[self.person]

    def update(self):
        return self.status

class Condition_PersonFound(py_trees.behaviour.Behaviour):
    def __init__(self, node, name="LookForPerson"):
        super().__init__(name)
        self.node = node
        self.subtree = self._build_subtree()

    def _build_subtree(self):
        look = LookAround(self.node)
        invert = py_trees.decorators.Inverter(name="Invert", child=look)
        detect = Condition_PersonDetected(self.node)

        policy=py_trees.common.ParallelPolicy.SuccessOnSelected(children=[detect])
        node = py_trees.composites.Parallel(name="Parallel", policy=policy)
        node.add_children([invert, detect])

        return node

    def initialise(self):
        self.subtree.tick_once()

    def update(self):
        self.subtree.tick_once()
        status = self.subtree.status
        return status

    def terminate(self, new_status):
        self.subtree.stop(new_status)

# notification
class Notify(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, text: str, name="Notify"):
        super().__init__(name)
        self.node = node
        self.text = text
        self._publisher = self.node.create_publisher(String, '/notify', 10)
        self.marker_publisher = self.node.create_publisher(Marker, '/my_text_marker', 10)
        self.marker_id = 0
        self.timer = None

    def initialise(self):
        self.node.get_logger().info(f"Notifying: {self.text}")

        msg = String()
        msg.data = self.text
        self._publisher.publish(msg)

        now = self.node.get_clock().now().to_msg()

        current_coords = Config.rooms.get(Config.current_room.value, (0.0, 0.0))

        # TEXT marker
        fg_marker = Marker()
        fg_marker.header.frame_id = "map"
        fg_marker.header.stamp = now
        fg_marker.ns = "text_marker"
        fg_marker.id = 1
        fg_marker.type = Marker.TEXT_VIEW_FACING
        fg_marker.action = Marker.ADD
        fg_marker.pose.position.x = current_coords[0]
        fg_marker.pose.position.y = current_coords[1]
        fg_marker.pose.position.z = 1.0  # just above background
        fg_marker.pose.orientation.w = 1.0
        fg_marker.scale.z = 0.4
        fg_marker.color.a = 1.0
        wrapped_text = textwrap.fill(self.text, width=45)
        fg_marker.text = wrapped_text

        lines = wrapped_text.count('\n') + 1

        # CUBE background marker
        bg_marker = Marker()
        bg_marker.header.frame_id = "map"
        bg_marker.header.stamp = now
        bg_marker.ns = "text_marker"
        bg_marker.id = 0
        bg_marker.type = Marker.CUBE
        bg_marker.action = Marker.ADD
        bg_marker.pose.position.x = current_coords[0]
        bg_marker.pose.position.y = current_coords[1]
        bg_marker.pose.position.z = 0.0  # slightly behind text
        bg_marker.pose.orientation.w = 1.0
        bg_marker.scale.x = 0.5 * lines  # height
        bg_marker.scale.y = 10.0  # width
        bg_marker.scale.z = 0.4  # depth
        bg_marker.color.r = 1.0
        bg_marker.color.g = 1.0
        bg_marker.color.b = 1.0
        bg_marker.color.a = 1.0  # semi-transparent
        # Publish both
        self.marker_publisher.publish(bg_marker)
        self.marker_publisher.publish(fg_marker)

        # Set up timer to remove both after 3 seconds
        self.timer = self.node.create_timer(3.0, self._remove_markers)

    def _remove_markers(self):
        for marker_id in [0, 1]:  # Background and text
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.node.get_clock().now().to_msg()
            marker.ns = "text_marker"
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.marker_publisher.publish(marker)

        if self.timer:
            self.timer.cancel()
            self.node.destroy_timer(self.timer)
            self.timer = None

    def update(self):
        if self.timer:
            # If the timer is still active, we are still "notifying"
            return py_trees.common.Status.RUNNING
        return py_trees.common.Status.SUCCESS


# HELPER NODES

class Wait(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="Wait"):
        super(Wait, self).__init__(name)
        self.node = node

    def update(self):
        return py_trees.common.Status.RUNNING

class SetTarget(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, target: str, name="SetTarget"):
        super().__init__(name)
        self.node = node
        self.target = target

    def initialise(self):
        self.node.get_logger().info(f"Setting target to {self.target}")
        Config.current_target = self.target

    def update(self):
        return py_trees.common.Status.SUCCESS

class LookAround(py_trees.behaviour.Behaviour):
    def __init__(self, node: Node, name="LookAround"):
        super().__init__(name)
        self.node = node
        self.angular_speed = 0.5  # radians per second
        self.duration = 12.56 / self.angular_speed
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
            print("LookAround: Duration reached, stopping.")
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status):
        if self._timer is not None:
            self._timer.cancel()
        stop = Twist()
        self._cmd_vel_pub.publish(stop)  

class LookAt(py_trees.behaviour.Behaviour):
    def __init__(self, node, name="LookAt", pos=0.0):
        super().__init__(name)
        self.node = node
        self._client = ActionClient(node, FollowJointTrajectory, '/head_controller/follow_joint_trajectory')
        self._goal_handle = None
        self._done = False
        self.pos = pos

    def initialise(self):
        self.node.get_logger().info(f"Initialising LookAt at pos={self.pos}")
        self._done = False
        if not self._client.wait_for_server(timeout_sec=2.0):
            self.node.get_logger().error("Head controller action server not available")
            self._done = True
            return

        from trajectory_msgs.msg import JointTrajectory
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = ['head_1_joint', 'head_2_joint']
        goal_msg.trajectory.points = [
            JointTrajectoryPoint(
                positions=[self.pos, 0.0],
                time_from_start=Duration(sec=2)
            ),
        ]

        send_goal_future = self._client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self._goal_response_callback)

    def _goal_response_callback(self, future):
        self.node.get_logger().info("Goal response callback triggered")
        self._goal_handle = future.result()
        if not self._goal_handle.accepted:
            self.node.get_logger().warn("Goal rejected")
            self._done = True
            return

        self.node.get_logger().info("Goal accepted, waiting for result...")
        self._result_future = self._goal_handle.get_result_async()
        self._result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        result = future.result().result
        self.node.get_logger().info(f"LookAt(pos={self.pos}) action complete")
        self._done = True


    def update(self):
        self.node.get_logger().info(f"Updating LookAt at pos={self.pos} done={self._done}")
        if self._done:
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
            return py_trees.common.Status.RUNNING
        return py_trees.common.Status.SUCCESS
    
    def person_callback(self, msg: Bool):
        self.person_detected = msg.data

# class Explore(py_trees.behaviour.Behaviour):
#     def __init__(self, node, name="Explore"):
#         super().__init__(name)
#         self.node = node
#         self.subtree = self._build_subtree()

#     def _build_subtree(self):
#         node = py_trees.composites.Sequence(name="ExploreSequence", memory=True)
#         for room in Config.Room:
#             if room != Config.current_room:
#                 navigate = NavigateTo(self.node, room_id=room)
#                 node.add_child(navigate)
#                 node.add_child(LookAround(self.node))
#         return node

#     def initialise(self):
#         self.subtree.tick_once()

#     def update(self):
#         self.subtree.tick_once()
#         status = self.subtree.status
#         return status

#     def terminate(self, new_status):
#         self.subtree.stop(new_status)
