import rclpy
from rclpy.node import Node
import py_trees
from bt_node.langchain_planner import get_plan
import json
from bt_node.bt_nodes import MoveForward, LookAround, Condition_PersonDetected, NavigateTo, Wait
from bt_node.legacy_nodes import RotateLeft, TurnAround, ChangePenColor, Condition_CheckInBounds
from bt_node.config import rooms, Room

class BehaviorTreeRoot(Node):
    def __init__(self):
        super().__init__('bt_node')
        self.bt = self.create_behavior_tree()
        self.get_logger().info("Behavior Tree Initialized")
        self.timer = self.create_timer(1.0, self.tick_tree)

    def create_behavior_tree(self):
        # root = py_trees.composites.Selector(name="RootSelector", memory=True)

        root = py_trees.composites.Sequence(name="Sequence", memory=True)

        look = LookAround(self)
        detect = Condition_PersonDetected(self)

        policy=py_trees.common.ParallelPolicy.SuccessOnSelected(children=[detect])
        node = py_trees.composites.Parallel(name="Parallel", policy=policy)
        node.add_children([look, detect])

        root.add_child(node)
        root.add_child(NavigateTo(self, x=7.0, y=7.0))

        # for roomidx, coords in rooms.items():
        #     root.add_child(NavigateTo())


        return py_trees.trees.BehaviourTree(root)

    def create_condition_node(self, condition_type):
        if condition_type == "Condition_CheckInBounds":
            return Condition_CheckInBounds(self)
        elif condition_type == "Condition_PersonDetected":
            return Condition_PersonDetected(self)
        
        return Condition_PersonDetected(self)
    
    def create_action_node(self, action_type, args=None):

        if action_type == "NavigateTo" and args is not None and len(args) == 1:
            room_name = args[0]
            coords = rooms[Room[room_name].value]
            print(coords)
            return NavigateTo(self, x=coords[1], y=coords[0])
        elif action_type == "TurnAround":
            return TurnAround(self)
        elif action_type == "MoveForward":
            return MoveForward(self)
        elif action_type == "RotateLeft":
            return RotateLeft(self)
        elif action_type == "LookAround":
            return LookAround(self)
        elif action_type == "ChangePenColor" and args is not None and len(args) == 3:
            return ChangePenColor(self, args[0], args[1], args[2])
        
        return Wait(self)

    def bcreate_behavior_tree(self):
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
