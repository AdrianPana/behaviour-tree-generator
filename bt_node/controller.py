from bt_node.langchain_planner import get_plan
from bt_node.behaviour_tree import BehaviorTreeRoot
import rclpy
from bt_node.config import Config

def main(args=None):
    print("Hi! How may I assist you today?")

    while True:
        user_input = input("Prompt: ")

        plan = None
        if user_input.startswith("test"):
            plan = None
        elif user_input.startswith("config"):
            print("Current configuration:")
            print(Config.people_rooms)
            print("Change people's locations in the building")

            person = input("Who do you want to move? (Alice/Bob): ")
            for room in Config.Room:
                print(f"{room.name} = {room.value}")

            room_id = input("Where do you want to move them? (0-6): ")
            # find enum element with value = 6
            
            room = list(Config.Room)[int(room_id)]
            for k, v in Config.people_rooms.items():
                if v == person:
                    del Config.people_rooms[k]
                    Config.people_rooms[room] = person
                    print(f"Moved {person} to {room.name}")
                    break
            continue
        else:
            plan = get_plan(user_input)

        rclpy.init(args=args)
        node = BehaviorTreeRoot(plan=plan)

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if getattr(node, 'done', False):
                break

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
