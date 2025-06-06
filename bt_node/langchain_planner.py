from langchain_ollama import ChatOllama
# from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableMap
from bt_node.config import Config
import os
from dotenv import load_dotenv
# Replace with your OpenAI key stored in .env file

# load_dotenv()  # Load environment variables from .env file
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0.0,)
llm = ChatOllama(model="mistral", temperature=0)

template="""
    You are a robot planner. Given the objective below, you must generate a plan that controls the robot.

    ## Rules:
    {rules}
    Try to use the context and your capabilities to achieve the goal in a logical way.
    
    ## Actions
    The robot can perform **only** the following actions:
    {actions}

    # Examples:
    {examples}
    # Task:
       
    ## Context
    Current context:
    {context}

    Here are the available rooms in the environment:
    {rooms}

    The only people in the environment are:
    {people}
    There's at most one person in a room at a time.

    ## Objective
    Objective: {situation}

    If you need an explanation on how to achieve the objective, only output: "HELP" and nothing else.

    {self_feedback}
    """
 
# context = "You are in the Hall."
 
rooms = """
    - HALL
    - OFFICE
    - DINING_ROOM
    - CLASSROOM1
    - CLASSROOM2
    - BATHROOM
    - LOUNGE
    """

people = """
    - Alice
    - Bob
    """

actions = """
    - NavigateTo(ROOM) # Sends the robot to the specified room.
    - Notify(MESSAGE) # Speaks the given message out loud. 
    - ExploreFindPerson(NAME) # Explores all the rooms in the environment until it finds the person with the given name.
      This action incorporates navigation and person detection and recognition, but it's quite inefficient.
      If you have information about the person's location, use `NavigateTo` and `Condition_PersonFound` instead.
      Use this action when you don't know how to find the person or you find that the information you had is wrong.
      No extra condition nodes needed.
    - Condition_PersonFound # Looks around the room for any person and returns SUCCESS if any person is found, FAILURE otherwise.
    - Condition_FaceRecognized(NAME) # Checks if the person is the one with the given name. Returns SUCCESS if the person is recognized, FAILURE otherwise.
    """
   # - Condition_PersonFound # Returns SUCCESS if the robot detects a person in front of him.

rules = """
    - Only use the exact actions listed below. **Do not invent, assume, or guess any new actions.**
    - It is not mandatory to use all the actions, strictly use the ones that are necessary to achieve the goal. Don't add extra functionality.
    - The actions have the **exact** effects as listed in the prompt. Do not assume any additional effects or side effects.
    - If an action is **not** explicitly listed in the available actions, it **must not** appear in the output. This includes helper functions, sensors, or derived logic.
    - You may use `if/else` statements to control flow if you see fit, but you may **not** use single-branch `if` statements, loops (e.g. `while`, `for`) or jumps (e.g., `break`, `continue`, `goto`).
    - If branches only check for success, if you want to tackle a failure case, use the `else` branch of the statement.
    - If statements should check one condition at a time, `and` and `or` are not allowed.
    - If you are unsure how to proceed without unlisted actions, **output nothing**. Do not attempt to create or improvise new actions.
    - The output must contain **only** the plan, with no additional text or explanation, in a pseudocode format.
    """

examples = """
    <example, result="invalid", reason="uses unlisted action `MoveFroward`">
    ## Input:
    ### Context: You are in the Office.
    ### Objective: Go to the Hall.
    ## Output:
        NavigateTo(HALL)
        MoveForward
    </example>
    
    <example, result="invalid", reason="uses unnecessary actions `Notify` and `Condition_PersonFound``">
    ## Input:
    ### Context: You are in the Office.
    ### Objective: Go to the Hall.
    ## Output:
        NavigateTo(HALL)
        if Condition_PersonFound
            Notify("Found you")
        else
            Notify("Hall is empty")
    </example>

    <example, result="invalid", reason="if statement checks for FAILURE instead of SUCCESS">
    ## Input:
    ### Context: You are in the Dining Room.
    ### Objective: If you find someone here, go to the Lounge. Otherwise, go to the Hall.
    ## Output:
        if Condition_PersonFound == FAILURE
            NavigateTo(LOUNGE)
        else
            NavigateTo(HALL)
    </example>

    <example, result="invalid", reason="uses context wrong, looks for Bob in the office, when you remember he was elsewhere">
    ## Input:
    ### Context: You are in the Hall. You remember that Alice is in the Office and Bob is in the Dining Room.
    ### Objective: find bob.
    ## Output:
        NavigateTo(OFFICE)
        if Condition_PersonFound
            if Condition_FaceRecognized("Bob")
                Notify("I found you, Bob!")
        else ExploreFindPerson("Bob")

    </example>

    <example, result="valid", reason="uses action `ExploreFindPerson` when no information is available">
    ## Input:
    ### Context: You are in the Dining Room.
    ### Objective: Tell Bob to hydrate.
    ## Output:
        ExploreFindPerson("Bob")
        Notify("Don't forget to hidrate!")
    </example>

    <example, result="valid", reason="uses action `ExploreFindPerson` as a last resort, when given information is not enough to find the person">
    ## Input:
    ### Context: You are in the Dining Room. You remember that Bob is in the Lounge.
    ### Objective: Tell Bob to hydrate.
    ## Output:
        NavigateTo(LOUNGE)
        if Condition_PersonFound
            if Condition_FaceRecognized("Bob")
                Notify("Don't forget to hidrate!")
        else ExploreFindPerson("Bob")
             Notify("Don't forget to hidrate!")
    </example>

    <example, result="valid", reason="uses only necessary action `ExploreFindPerson``">
    ## Input:
    ### Context: You are in the Office.
    ### Objective: Go to Alice.
    ## Output:
        ExploreFindPerson("Alice")
    </example>

    <example, result="valid", reason="uses only listed actions and room names, if statement checks for SUCCESS">
    ## Input:
    ### Context: You are in the Hall.
    ### Objective: Look for Alice in the Bathroom. If you find Alice, remind them to hydrate.
    ## Output:
        NavigateTo(BATHROOM)
        if Condition_PersonFound
            if Condition_FaceRecognized("Alice")
                Notify("Don't forget to hidrate!")
    </example>
    """

feedback_template = """
    I have a generated plan that should control a robot in order to achieve a given objective.

    ## Context
    Current context:
    {context}

    Here are the available rooms in the environment:
    {rooms}

    The only people in the environment are:
    {people}
    There's at most one person in a room at a time.

    ## Objective
    Objective: {situation}

    # Plan
    This is the plan that was generated:
    {plan}

    Given the rules: 
    {rules}

    Provide feedback on the generated plan. If you find something wrong with it, please explain why it is wrong and give suggestions on how to correct it.
    If you find the plan satisfactory, please only return the string "VALID" as the only output, without any additional text.
    """

plan_to_json_template = """
    Using a generated plan given below, convert it into a JSON format.

    ## Rules:
    Each JSON element should be composed of a type (the name of the action or condition), a true key (for the if brach) and a false key (for the else branch).
    The contents of the true and false keys are lists built recursively just like the root node. If any of the lists are empty, omit the key. Do not provide empty objects (so no `{{}}`).
    The json should have one single object as a root, not a list. If there is no suitable action/condition/node to start as a root, use a "type" key set to "Sequence", and put all the actions that are on the same level in the "true" key.
    I only need you to parse the plan and convert from pseudocode to JSON. You don't have to do any other processing or logic, don't add any additional actions and don't remove anything.
    Please make sure to use the same names for the nodes as in the plan.
    You may only have "type", "true", "false" and "args" keys in the JSON objects. **Do not invent, assume, or guess any new keys.** If a key is **not** explicitly listed in "type", "true", "false" and "args", it **must not** appear in the output.
    Output only the converted JSON, no other text or comments, not even pointing out that this is a JSON.
    Remove any comments. they are not allowed in a JSON file and will break my parsing algorithm.

    # Examples

    <example, result="invalid", reason="provides a list as a root for the JSON">
    ## Input:
    NavigateTo(BATHROOM)
    if Condition_PersonFound
        Notify("Don't forget to hidrate!")
    else
        NavigateTo(HALL)

    ## Output:
    {{
    [
        {{
        "type": "NavigateTo",
        "args": ["BATHROOM"]
        }},
        {{
        "type": "Condition_PersonFound",
        "true": [
            {{
                "type": "Notify",
                "args": ["Don't forget to hidrate!"]
            }}
            ],
        "false": [
            {{
                "type": "NavigateTo",
                "args": [LOUNGE]
            }}
        ]
        }}
    ]
    }}
    </example>

    <example, result="valid">
    ## Input:
    NavigateTo(LOUNGE)
    if Condition_PersonFound
        if Condition_FaceRecognized("Bob")
            Notify("Don't forget to hidrate!")
    else ExploreFindPerson("Bob")
            Notify("Don't forget to hidrate!")

    ## Output:
    {{
    "type": "Sequence",
    "true": [
        {{
        "type": "NavigateTo",
        "args": ["LOUNGE"]
        }},
        {{
        "type": "Condition_PersonFound",
        "true": [
            {{
                "type": "Condition_FaceRecognized",
                "args": ["Bob"],
                "true": [
                    {{
                        "type": "Notify",
                        "args": ["Don't forget to hidrate!"]
                    }}
                ]
            }}
            ],
        "false": [
            {{
                "type": "ExploreFindPerson",
                "args": ["Bob"]
            }},
            {{
                "type": "Notify",
                "args": ["Don't forget to hidrate!"]
            }}
        ]
        }}
    ]
    }}
    </example>

    # Task:
    ## Plan
    {plan}
    """

def get_plan(situation = "Go through the house until you find someone. If you find someone, come back here and tell me in which room you found them."):

    context = "You are in the " + Config.current_room.name + ". "
    if Config.people_room_memory:
        context += "You remember that: "
        context += ", ".join([(f"{k} was in the {v.name}" if v else f"You don't know where {k} was") for k, v in Config.people_room_memory.items()])

    self_feedback = ""

    prompt = PromptTemplate(input_variables=["situation", "actions", "rooms", "context", "people", "rules", "self_feedback", "examples"], 
        template=template)

    feedback_prompt = PromptTemplate(input_variables=["situation", "context", "people", "rules", "plan"],
        template=feedback_template)

    json_prompt = PromptTemplate(template=plan_to_json_template, input_variables=["plan"])
    
    plan_chain = prompt | llm | RunnableLambda(lambda plan: {"plan": plan})
    feedback_chain = feedback_prompt | llm
    json_chain = json_prompt | llm

    def apply_feedback(inputs):
        plan = inputs["plan"]
        print(f"Plan: {plan}")
        feedback = feedback_chain.invoke({"plan": plan, **inputs})
        return {"plan": plan, "feedback": feedback}

    # Test the decision
    chain = prompt | llm
    plan = chain.invoke({
        "situation": situation,
        "actions": actions,
        "rooms": rooms,
        "context": context,
        "people": people,
        "rules": rules,
        "examples": examples,
        "self_feedback": ""
    }).content

    # chain = feedback_prompt | llm
    # feedback = chain.invoke({
    #     "situation": situation,
    #     "context": context,
    #     "people": people,
    #     "rules": rules,
    #     "rooms": rooms,
    #     "plan": plan,
    # }).content

    # print(feedback)
    return plan

    self_feedback_chain = (
        RunnableMap({
        "situation": RunnablePassthrough(),
        "actions": RunnablePassthrough(),
        "rooms": RunnablePassthrough(),
        "context": RunnablePassthrough(),
        "people": RunnablePassthrough(),
        "rules": RunnablePassthrough(),
        "examples": RunnablePassthrough(),
        "self_feedback": RunnablePassthrough(),
        })
        | plan_chain
        | RunnableLambda(apply_feedback)
    )

    while True:
        result = self_feedback_chain.invoke({
            "situation": situation,
            "actions": actions,
            "rooms": rooms,
            "context": context,
            "people": people,
            "rules": rules,
            "examples": examples,
            "self_feedback": self_feedback
        }).content

        plan = result["plan"]
        feedback = result["feedback"]

        print(plan)
        print(feedback)


        if feedback.startswith("VALID"):
            final_output = json_chain.invoke({"plan": plan})
            break
        else:
            self_feedback= """
            You already generated this plan for the objective:
            """ + plan + """

            You have received feedback on it:
            """ + feedback + """

            Use the feedback to improve the plan and return the improved plan in the same format as before.
            """

    return final_output

if __name__ == "__main__":
    prompt = input("Situation: ")
    plan = get_plan(prompt)
    print(f"LLM Plan:\n {plan}")
