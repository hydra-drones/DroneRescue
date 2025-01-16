from typing import Dict, Any, Literal
from langchain.schema import HumanMessage, SystemMessage
from v2.interfaces.agent_schema import AgentRunnable
from v2.states.agent_state import AgentState


class DecisionAgent(AgentRunnable):
    def __init__(
        self, llm_api_key: str, common_agent_state: AgentState, llm_model: str = "gpt-4"
    ):
        super().__init__(llm_api_key, common_agent_state, llm_model)
        self.common_agent_state = common_agent_state

    def invoke(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Process a single input and return a result.
        Input format: {"previous_state": str}
        Output format: {"response": str}
        """

        current_state = args[0]
        system_prompt = (
            "CURRENT STRATEGY:\n"
            + str(current_state.get("strategy"))
            + "\n\n"
            + "TEAMMATE AGENTS:\n"
            + str(current_state.get("teammate_agent_info"))
            + "\n\n"
            + "PREVIOUS MESSAGE:\n"
            + str(current_state.get("messages")[-1:])
            + "\n\n"
            "STATE DESCRIPTION:\n"
            + str(current_state.get("state_description"))
            + "\n\n"
            + f"""Your task is to analyze the provided state description and the current strategy
            to determine the next best step. You have the following possible options:
            1) Call the Action Agent (e.g., to perform a specific task or move),
            2) Call the Sending Message to Teammate Agent (e.g., to share critical information),
            3) Change the Current Strategy Agent (e.g., if a strategic shift is needed).

            **OUTPUT**:
            - "message": A concise message that the next agent you choose will read.
            This message should include all relevant details or instructions needed for that agent.
            - "next": The identifier of the next agent to call. It must be one of: {current_state.get('next_agent')}

            **CONSIDERATIONS**:
            - Leverage all relevant details in the state description (e.g., threats, opportunities, teammate requests).
            - Ensure your decision aligns with or appropriately revises the current strategy.
            - If communicating with teammates, provide only necessary, actionable information.
            - If calling the action agent, specify the intended action or context.
            - If changing strategy, clearly state why.

            **TASK**:
            The "message" must capture the essential details needed by the called agent,
            and "next" must specify which agent should be invoked.

            **EXAMPLE RESPONSE**:

            1. Calling the Action Agent:
            "message": "We should move to sector 5 to investigate a high-priority signal.",
            "next": <take relevant agent from the list given in user prompt>

            2. Sending a message to a teammate:
            "message": "Teammate 002, I'm en route to sector 5. Please confirm your location and status.",
            "next": <take relevant agent from the list given in user prompt>

            3. Changing the strategy:
            "message": "Given the battery constraints and enemy activity, we need to switch to a defensive stance in the northern sectors.",
            "next": <take relevant agent from the list given in user prompt>
            """
        )

        human_prompt = f"""
        Please review the state description and another information about the agent you've been given.
        Based on this information, decide the best immediate step.
        You may choose one of the following:
        1) Call the Action Agent (e.g., to perform an action like moving or investigating),
        2) Send a message to a teammate,
        3) Change the current strategy.

        Provide your decision as:
        - A "message" key: describing any details or instructions for the agent you select.
        - A "next" key: specifying which agent to call ({current_state.get('next_agent')}).

        Note: choosing "action" agent you will able to use another tools only on the next step. So, if it's needed, use the another tools first, onyl then "action" agent.

        Please ensure that your message is clear, actionable, and consistent with the situation and strategy.
        """

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]

        # Dynamically create structure output
        allowed_next_agents = tuple(current_state.get("next_agent"))
        model_fields = {
            "message": (str, ...),
            "next": (Literal[allowed_next_agents], ...),
        }
        DecisionMakingOutput = self.create_structure_output(
            "DecisionMakingOutput", model_fields
        )

        # Inference model
        structured_llm = self.llm.with_structured_output(DecisionMakingOutput)
        response = structured_llm.invoke(messages)

        message = response.message
        next_agent = response.next

        if current_state.get("verbose"):
            print(
                f"Decision Maker\nMessage {message}\nNext agent : {next_agent}\n"
                + "-" * 50
            )

        return {"messages": [message], "next": next_agent}
