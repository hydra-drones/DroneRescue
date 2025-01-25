from typing import Dict, Any, Literal
from langchain.schema import HumanMessage, SystemMessage
from v2.interfaces.agent_schema import AgentRunnable
from v2.states.agent_state import AgentState


class CommunicationAgent(AgentRunnable):
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
            + "TEAMMATE AGENT INFORMATION:\n"
            + str(current_state.get("teammate_agent_info"))
            + "\n\n"
            + """Your task is to:
            1. Review the latest instructions or requests from the Decision Maker.
            2. Refer to the relevant teammate agent information (such as agent IDs, roles, positions, etc.).
            3. Compose a clear, concise message that accurately conveys the Decision Maker’s intent to the appropriate teammate(s).

            **GUIDELINES**:
            - Make sure you address the correct teammate agent based on the provided info and the Decision Maker’s message.
            - Include enough context so the recipient understands what is being asked or shared.
            - If any details are unclear, make a reasonable assumption or ask for clarification within the message.

            **OUTPUT FORMAT**:
            "recipient": "<ID or name of the agent you are messaging>",
            "message": "<the composed message text>"

            Example:
            "recipient": "002",
            "message": "We need you to secure sector 5. Please confirm your battery level and position."


            Keep the message direct, actionable, and consistent with the Decision Maker’s instructions and the teammate info provided above.
            """
        )

        agent_ids_list = [
            key
            for agent_info in current_state.get("teammate_agent_info")
            for key in list(agent_info.keys())
        ]

        human_prompt = f"""
        You have been given the last Decision Maker message and the following teammate agent IDs:
        {agent_ids_list}
        Please analyze the Decision Maker’s instructions and decide which agent(s) need to receive the message. Then compose a concise, actionable message.
        "recipient": "<ID or name of the agent you are messaging>",
        "message": "<the composed message text>"
        Be sure to use an agent ID from the list: {agent_ids_list}.
        """

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]

        # Dynamically create structure output
        allowed_agents = tuple(agent_ids_list)
        model_fields = {
            "message": (str, ...),
            "recipient": (Literal[allowed_agents], ...),
        }

        CommunicatorOutput = self.create_structure_output(
            "CommunicatorOutput", model_fields
        )

        # Inference model
        structured_llm = self.llm.with_structured_output(CommunicatorOutput)
        response = structured_llm.invoke(messages)

        message = response.message
        recipient = response.recipient

        if current_state.get("verbose"):
            print(f"Communicator\nID:{recipient}\nMessage: {message}\n" + "-" * 50)

        # Update messages
        new_message = f"Message '{message}' has been sent to the {recipient} agent."
        messages_history = self.common_agent_state.get("messages")
        if len(messages_history) > 5:
            messages_history.pop(0)
            messages_history.append(new_message)
        else:
            messages_history.append(new_message)

        return {
            "message_to_teammate_agent": [(recipient, message)],
            "messages": messages_history,
        }
