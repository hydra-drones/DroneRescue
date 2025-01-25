from typing import Dict, Any, Literal
from langchain.schema import HumanMessage, SystemMessage
from v2.interfaces.agent_schema import AgentRunnable
from v2.states.agent_state import AgentState


class ActionAgent(AgentRunnable):
    def __init__(self, llm, common_agent_state: AgentState):
        super().__init__(common_agent_state)
        self.common_agent_state = common_agent_state
        self.llm = llm

    def invoke(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Process a single input and return a result.
        Input format: {"previous_state": str}
        Output format: {"response": str}
        """

        current_state = args[0]
        system_prompt = (
            "LAST DECISION MAKER MESSAGE:\n"
            + str(current_state.get("messages")[-1])
            + "\n\n"
            + "ACTION HISTORY:\n"
            + str(current_state.get("action_history")[-2:])
            + "\n\n"
            + "SPEED HISTORY:\n"
            + str(current_state.get("speed_history")[-2:])
            + "\n\n"
            + "CURRENT SECTOR:\n"
            + str(current_state.get("current_sector"))
            + "\n\n"
            + "MAP OF SECTORS:\n"
            + str(current_state.get("map_of_sectors"))
            + "\n\n"
            + "ACTUAL POSITION:\n"
            + str(current_state.get("current_position"))
            + "Remember that position in the Y, X format."
            + "\n\n"
            + "ENVIRONMENT SIZE:\n"
            + str(current_state.get("env_size"))
            + "\n\n"
            + f"""Your task is to select and describe the next action for this agent.
            Please follow these guidelines:

            1. **Possible Actions** (example set):
            - 0: Move upward.
            - 1: Move downward
            - 2: Move left
            - 3: Move right
            - 4: wait, temporarily hold the position.

            2. **Speed Selection**:
            - You may choose from the following speeds: {current_state.get('speeds')}.
            - Higher speed may be selected if the target or critical sector is far away.
            - Lower speed can be used for caution or to conserve energy.
            - Speed 0 is equivalent to stay in place

            3. **Considerations**:
            - Refer to the last message from the Decision Maker to understand immediate instructions or constraints.
            - Check the action history and speed history to avoid repeating unproductive patterns.
            - Use the current sector and the sector map to decide a valid movement or investigation target.
            - If the agent needs to move far to reach a potential target, favor a higher speed, unless battery or safety concerns suggest otherwise.
            - Remember, that the environment uses a non-standard coordinate system where the origin (0,0) is in the top-left corner, and the Y-coordinate increases as the agent moves downward (contrary to the conventional Cartesian system where Y increases upward).

            Carefully evaluate the distances, priorities, and the Decision Maker’s instructions
            before finalizing your choice of action and speed.

            ### Important
            - All coordinates provided in a prompt follow the structure (Y, X). For example, the coordinate (1, 29) means Y=1 and X=29. Note that the Y value increases as you move downward
            """
        )

        human_prompt = f"""
        You have been given the last message from the Decision Maker, as well as your action history, speed history, current sector, and the map of sectors.
        You can choose your next action from the following list:
        {current_state.get("actions")}
        And you can select your speed from the following list:
        {current_state.get("speeds")}

        Please analyze all the information and decide on the best action and speed.
        NOTE : remember, that if you current position contains some border value, like (3,0) - zero means that you are near border of the area
        and you should go in to opposite direction, to (3,4) for example. Environment size : {current_state.get("env_size")}
        Ensure that your chosen action and speed align with the situation described by the Decision Maker and your own recent history.
        """

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]

        # Dynamically create structure output
        allowed_actions = tuple(current_state.get("actions"))
        allowed_speeds = tuple(current_state.get("speeds"))
        model_fields = {
            "action": (Literal[allowed_actions], ...),
            "speed": (Literal[allowed_speeds], ...),
        }

        ActionOutput = self.create_structure_output("ActionOutput", model_fields)

        # Inference model
        structured_llm = self.llm.with_structured_output(ActionOutput)
        response = structured_llm.invoke(messages)

        action = response.action
        speed = response.speed

        if current_state.get("verbose"):
            print(f"Action Agent\nAction {action}\nSpeed : {speed}\n" + "-" * 50)

        # Update Action
        action = str(action)
        action_history = self.common_agent_state.get("action_history")
        if len(action_history) > 5:
            action_history.pop(0)
            action_history.append(action)
        else:
            action_history.append(action)

        # Update Speed
        speed = str(speed)
        speed_history = self.common_agent_state.get("speed_history")
        if len(speed_history) > 5:
            speed_history.pop(0)
            speed_history.append(speed)
        else:
            speed_history.append(speed)

        return {"action_history": action_history, "speed_history": speed_history}
