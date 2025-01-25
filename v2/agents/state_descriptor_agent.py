from typing import Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from v2.interfaces.agent_schema import AgentRunnable
from v2.states.agent_state import AgentState


class StateDescriptor(AgentRunnable):
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

        system_prompt = (
            "BATTERY LEVEL:\n"
            + str(self.common_agent_state.get("battery_level"))
            + "\n\n"
            + "MAP OF SECTORS:\n"
            + str(self.common_agent_state.get("map_of_sectors"))
            + "\n\n"
            + "ACTION HISTORY:\n"
            + str(self.common_agent_state.get("action_history")[-5:])
            + "\n\n"
            + "STRATEGY:\n"
            + str(self.common_agent_state.get("strategy"))
            + "\n\n"
            + "AVAILABLE TEAMMATE AGENTS:\n"
            + str(self.common_agent_state.get("teammate_agent_info"))
            + "\n\n"
            + "MESSAGES FROM AGENTS:\n"
            + str(self.common_agent_state.get("messages_from_agents"))
            + "\n\n"
            + "OBSERVATION (LOCAL):\n"
            + "Observation describes objects what agent sees. Each value in the observation list symbolizes some object. Here is the mapping:"
            + str({"BACKGROUND": 0, "TARGET_POINT": 2, "AGENT_POSITION": 4})
            + str(self.common_agent_state.get("observation"))
            + "\n\n"
            + "CURRENT POSIITION (GLOBAL MAP):\n"
            + str(self.common_agent_state.get("current_position"))
            + "\n\n"
            + "SECTORS WHERE TARGETS CAN POTENTIALLY BE:\nHere each tuple represents the area with potential target location. You can choose one of them"
            + str(self.common_agent_state.get("areas_of_potential_target_locations"))
            + "\n\n"
            + "CURRENT SECTOR:\n"
            + str(self.common_agent_state.get("current_sector"))
            + "\n\n"
            + """Your task is to analyze the above state information and produce a concise, clear descriptor of the current situation.
            Please summarize:
              - The most relevant details about the agent’s immediate environment (targets, sectors, teammates),
              - Any important constraints or notable conditions (battery level, past actions, messages),
              - How these factors might affect the next step’s decision.
              - Opportunities for collaboration with teammate agents.
            This summary will be used by the next agent to decide on an appropriate action.

            Make sure the description is logically structured and highlights only the key points that inform decision-making.
            """
        )

        human_prompt = f"""
          Please compare the new state information to the previous state description
          {self.common_agent_state.get('state_description')} and summarize **only the changes** that occurred.
          If nothing has changed, state that there are no significant updates.
          Keep your description concise and focused on details relevant to decision-making.
          Important : check if the target location in the observation area. Remember inform teammate agent about targets.
          """

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]

        response = self.llm.invoke(messages)
        description_message = response.content

        if self.common_agent_state.get("verbose"):
            print(f"State Descriptor\nMessage {description_message}\n" + "-" * 50)

        # Update messages
        messages_history = self.common_agent_state.get("messages")
        if len(messages_history) > 5:
            messages_history.pop(0)
            messages_history.append(description_message)
        else:
            messages_history.append(description_message)

        return {"messages": messages_history, "state_description": description_message}
