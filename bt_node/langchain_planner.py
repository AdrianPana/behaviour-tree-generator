from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from bt_node.config import Room
from langchain_core.runnables import RunnablePassthrough

# Replace with your OpenAI key or configure via env var
llm = ChatOllama(model="mistral", temperature=0)

template="""
    You are a robot planner. Given the following situation:
    {situation}

    You must generate a plan that controls the robot.
    Current context:
    {context}

    The robot can perform **only** the following actions:
    {actions}

    Here are the available rooms in the environment:
    {rooms}

    Rules:
    - Only use the exact actions listed above. **Do not invent, assume, or guess any new actions.**
    - The actions have the **exact** effects as listed in the prompt. Do not assume any additional effects or side effects.
    - If an action is **not** explicitly listed in `{actions}`, it **must not** appear in the output. This includes helper functions, sensors, or derived logic.
    - You may use `if/else` statements to control flow if you see fit, but you may **not** use single-branch `if` statements, loops (e.g. `while`, `for`) or jumps (e.g., `break`, `continue`, `goto`).
    - If statements should check one condition at a time, `and` and `or` are not allowed.
    - If you are unsure how to proceed without unlisted actions, **output nothing**. Do not attempt to create or improvise new actions.
    - The output must contain **only** the plan, with no additional text or explanation, in a pseudocode format.


    Examples:

    Invalid example (uses unlisted action `MoveForward`):
        NavigateTo(HALL)
        MoveForward
    
    Invalid example (uses unlisted room name `KITCHEN`):
        NavigateTo(KITCHEN)

    Valid example (uses only listed actions and room names):
        NavigateTo(HALL)
        """
 
context = "You are in the Hall."

situation = "Go to the classrooms and come back."
 
rooms = """
    - HALL
    - OFFICE
    - DINING_ROOM
    - CLASSROOM1
    - CLASSROOM2
    - BATHROOM
    - LOUNGE
    """

actions = """
    - NavigateTo(ROOM) # Sends the robot to the specified room.
    """
    # - RotateLeft # Rotates the robot by approximately 90 degrees to the left. This action only changes the robot's orientation, not its position.
    # - Condition_PersonDetected # Returns SUCCESS if the robot detects a person in front of him.
    # - Wait # Stops the robot when it has reached its goal.
    # - Condition_CheckInBounds # Returns SUCCESS if the robot's position is in bounds, FAILURE if out of bounds.
    # - RotateLeft # Rotates the robot to the left in place by 180 degrees. This action only changes the robot's orientation, not its position.
    # - ChangePenColorCyclic # Changes the trail color in a cyclic fashion from RED to GREEN to BLUE and to RED again and so on. This action only has a visual effect and does not affect the robot's position or orientation.
    # - ChangePenColor(r, g, b) # Changes the trail color to the specified RGB values given as parameters. This action only has a visual effect and does not affect the robot's position or orientation.
    # - Condition_PersonInRoom # Returns SUCCESS if there is a person in the room, FAILURE if there is not.
    # - Condition_TimeToTakeMedicine # Returns SUCCESS if it is time to take medicine, FAILURE if it is not.
    # - RemindToTakeMedicine # Reminds the person to take medicine. This action only has an effect if there is a person in the same room as the robot.


plan_to_json_template = """
    {plan}
    Using the generated plan, convert it into a JSON format.
    Each JSON element should be composed of a type (the name of the action or condition), a true key (for the if brach) and a false key (for the else branch).
    The contents of the true and false keys are lists built recursively just like the root node. If any of the lists are empty, omit the key. Do not provide empty objects (so no `{{}}`).
    The json should have one single object as a root, not a list. If there is no suitable action/condition/node to start as a root, use a "type" key set to "Sequence", and put all the actions that are on the same level in the "true" key.
    I only need you to parse the plan and convert from pseudocode to JSON. You don't have to do any other processing or logic, don't add any additional actions and don't remove anything.
    Please make sure to use the same names for the nodes as in the plan.
    You may only have "type", "true", "false" and "args" keys in the JSON objects. **Do not invent, assume, or guess any new keys.** If a key is **not** explicitly listed in "type", "true", "false" and "args", it **must not** appear in the output.
    Output only the converted JSON, no other text or comments, not even pointing out that this is a JSON.
    Remove any comments. they are not allowed in a JSON file and will break my parsing algorithm.

    For example:
        NavigateTo(HALL)

    Should be converted to:
    {{
    "type": "Sequence",
    "true": [
        {{
            "type": "NavigateTo",
            "args": [HALL]
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
    #     "actions": actions,
    #     "rooms": rooms,
    #     "context": context
    # }).content

    full_chain = (
        {
        "situation": RunnablePassthrough(),
        "actions": RunnablePassthrough(),
        "rooms": RunnablePassthrough(),
        "context": RunnablePassthrough()
        }
        | chain
        | (lambda plan: {"plan": plan} )
        | second_chain
    )

    plan = full_chain.invoke({
        "situation": situation,
        "actions": actions,
        "rooms": rooms,
        "context": context
    }).content

    return plan

if __name__ == "__main__":
    plan = get_plan()
    print(f"LLM Plan:\n {plan}")
