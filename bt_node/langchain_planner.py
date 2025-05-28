from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bt_robot.lisp_parser import parse_expr_to_dict
from langchain_core.runnables import RunnablePassthrough

# Replace with your OpenAI key or configure via env var
llm = ChatOllama(model="mistral", temperature=0)

template="""
    You are a robot planner. Given the following situation:
    {situation}

    You must generate a plan that controls the robot.

    The robot can perform **only** the following actions:
    {actions}

    Rules:
    - It is implied that the robot will execute this plan again and again. Do not include any phrases, comments, or notation indicating repetition, loops, or indefinite execution. Your output must be only the actions exactly as listed, with no additional explanation, comments, or repeated patterns.
    - Only use the exact actions listed above. **Do not invent, assume, or guess any new actions.**
    - The actions have the **exact** effects as listed in the prompt. Do not assume any additional effects or side effects.
    - If an action is **not** explicitly listed in `{actions}`, it **must not** appear in the output. This includes helper functions, sensors, or derived logic.
    - You may use `if/else` statements to control flow if you see fit, but you may **not** use single-branch `if` statements, loops (e.g. `while`, `for`) or jumps (e.g., `break`, `continue`, `goto`).
    - If statements should check one condition at a time, `and` and `or` are not allowed.
    - If you are unsure how to proceed without unlisted actions, **output nothing**. Do not attempt to create or improvise new actions.
    - The output must contain **only** the plan, with no additional text or explanation, in a pseudocode format.


    Examples:

    Invalid example (uses unlisted action `ScanArea`):
        ScanArea
        MoveForward

    Invalid example (mentions repetition explicitly):
        MoveForward
        RotateLeft
        ... (Repeat indefinitely)

    Valid example (uses only listed actions and does not mention repetition, also provides a stop):
        MoveForward
        RotateLeft
    
    Valid example (uses only listed actions, in this case we also use conditions):
        if Condition_PersonDetected
            MoveForward
        else
            RotateLeft
        """
 
situation = "Come sit next to me."

actions = """
    - MoveForward # Moves the robot forward a set distance.
    - RotateLeft # Rotates the robot by approximately 90 degrees to the left. This action only changes the robot's orientation, not its position.
    - Condition_PersonDetected # Returns SUCCESS if the robot detects a person in front of him.
    """
    # - Wait # Stops the robot when it has reached its goal.
    # - Condition_CheckInBounds # Returns SUCCESS if the robot's position is in bounds, FAILURE if out of bounds.
    # - RotateLeft # Rotates the robot to the left in place by 180 degrees. This action only changes the robot's orientation, not its position.
    # - ChangePenColorCyclic # Changes the trail color in a cyclic fashion from RED to GREEN to BLUE and to RED again and so on. This action only has a visual effect and does not affect the robot's position or orientation.
    # - ChangePenColor(r, g, b) # Changes the trail color to the specified RGB values given as parameters. This action only has a visual effect and does not affect the robot's position or orientation.
    # - Condition_HasKey # Returns SUCCESS if the robot has a key, FAILURE if it does not.
    # - Condition_DoorIsLocked # Returns SUCCESS if the door is locked, FAILURE if it is not.
    # - UnlockDoor # Unlocks a door if the robot has a key. This action only has an effect if the robot has a key and the door is locked.
    # - Condition_AtDoor # Returns SUCCESS if the robot is at a door, FAILURE if it is not.
    # - Condition_DoorInFront # Returns SUCCESS if there is a door in front of the robot at any distance from it, FAILURE if there is not.
    # - OpenDoor # Opens the door if the robot is at it. This action only has an effect if the robot is at a door and the door is not locked.
    # - Condition_DoorIsOpen # Returns SUCCESS if the door is open, FAILURE if it is not.
    # - Condition_PersonInRoom # Returns SUCCESS if there is a person in the room, FAILURE if there is not.
    # - Condition_TimeToTakeMedicine # Returns SUCCESS if it is time to take medicine, FAILURE if it is not.
    # - RemindToTakeMedicine # Reminds the person to take medicine. This action only has an effect if there is a person in the same room as the robot.
    # - EnterRoom # Enters the room if the robot is at the door and the door is open. This action only has an effect if the robot is at a door and the door is not open.

plan_to_json_template = """
    {plan}
    Using the generated plan, convert it into a JSON format.
    Each JSON element should be composed of a type (the name of the action or condition), a true key (for the if brach) and a false key (for the else branch).
    The contents of the true and false keys are lists built recursively just like the root node. If any of the lists are empty, omit the key. Do not provide empty objects (so no `{{}}`).
    If there is no suitable condition to start as a root, use a "type" key set to "Sequence", and put all the actions that are on the same level in the "true" key.
    I only need you to parse the plan and convert from pseudocode to JSON. You don't have to do any other processing or logic, don't add any additional actions and don't remove anything.
    Please make sure to use the same names for the nodes as in the plan.
    You may only have "type", "true", "false" and "args" keys in the JSON objects. **Do not invent, assume, or guess any new keys.** If a key is **not** explicitly listed in "type", "true", "false" and "args", it **must not** appear in the output.
    Output only the converted JSON, no other text or comments, not even pointing out that this is a JSON.
    Remove any comments. they are not allowed in a JSON file and will break my parsing algorithm.

    For example:
        if Condition_CheckInBounds
            ChangePenColor(0, 255, 0)
            MoveForward
        else
            RotateLeft
            MoveForward

    Should be converted to:
    {{
    "type": "Condition_CheckInBounds",
    "true": [
        {{
            "type": "ChangePenColor",
            "args": [0, 255, 0]
        }},
        {{
        "type": "MoveForward"
        }}
    ],
    "false": [
        {{
        "type": "RotateLeft"
        }},
        {{
        "type": "MoveForward"
        }}
    ]
    }}


    """

def get_plan():

    prompt = PromptTemplate(input_variables=["situation", "actions"], 
        template=template)

    yaml = PromptTemplate(template=plan_to_json_template, input_variables=["plan"])
    
    chain = prompt | llm
    second_chain = yaml | llm

    # # Test the decision
    # plan = chain.invoke({
    #     "situation": situation,
    #     "actions": actions
    # }).content

    full_chain = (
        {
        "situation": RunnablePassthrough(),
        "actions": RunnablePassthrough()
        }
        | chain
        | (lambda plan: {"plan": plan} )
        | second_chain
    )

    plan = full_chain.invoke({
        "situation": situation,
        "actions": actions
    }).content

    # plan = """
    #     {
    #     "type": "Selector",
    #     "name": "StayInBoundsOrTurn",
    #     "children": [
    #         {
    #             "type": "Sequence",
    #             "name": "MoveIfInBounds",
    #             "children": [
    #                 {
    #                     "type": "CheckInBounds"
    #                 },
    #                 {
    #                     "type": "MoveForward"
    #                 }
    #             ]
    #         },
    #         {
    #             "type": "Sequence",
    #             "name": "HandleOutOfBounds",
    #             "children": [
    #                 {
    #                     "type": "RotateLeft"
    #                 },
    #                 {
    #                     "type": "ChangePenColorCyclic"
    #                 },
    #                 {
    #                     "type": "MoveForward"
    #                 }
    #             ]
    #         }
    #     ]
    # }
    # """
    return plan

if __name__ == "__main__":
    plan = get_plan()
    print(f"LLM Plan:\n {plan}")

# code_prompt = ChatPromptTemplate.from_template("""
# You are a Python developer building a behavior tree using the `py_trees` library.

# Given this LLM plan:
# ---
# {plan}
# ---

# Write the Python code for a function called `create_behavior_tree(self)` that builds this behavior tree.
                                               
# Use the following predefined behavior tree nodes (you don't have to use all of them):

#     - MoveForward()
#     - CheckInBounds()
#     - TurnAround()
#     - RotateLeft()
#     - RotateRight()
#     - ChangePenColorCyclic()

# Use the py_trees composites `Sequence`, `Selector`. Include all necessary node instances like `MoveForward(self)`, and return a `py_trees.trees.BehaviourTree`.
# Remember, once the tree ticks to the end, it will start over by itself. Also, make sure the composites have a name and memory set to True.

# As an additional tip, whenever you see an if branch, the node that caused it should be put together with the SUCCESS branch as the first node of a Selector, and the other branch as a second node of the Selector.                                          

# Only output Python code, nothing else. Make sure the syntax is correct and the code is complete.
# """)

# code_chain = LLMChain(llm=llm, prompt=code_prompt)
# code = code_chain.run({"plan": plan})

# print("\nGenerated Behavior Tree Code:\n")
# print(code)

# Implementeaza noduri pentru turtle sim : check bounds, move forward, rotate, change pen color
# Lipeste nodurile ca sa simulezi un comportament, asigura-te ca e posibil
# Introdu LLM-ul si vezi daca poata sa decida aceiasi pasi