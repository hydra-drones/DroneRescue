from typing import Dict, Any, Literal
from langchain.schema import HumanMessage, SystemMessage
from v2.interfaces.agent_schema import AgentRunnable
from v2.states.agent_state import AgentState


class StrategyAgent(AgentRunnable):
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
            "PREVIOUS STATE:\n"
            + str(current_state.get("state_description"))
            + "\n\n"
            + "PREVIOUS DECISION MAKER MESSAGE:\n"
            + str(current_state.get("messages")[-1])
            + "\n\n"
            + "CURRENT STRATEGY:\n"
            + str(self.common_agent_state.get("strategy"))
            + "\n\n"
            + """Your task is to revise or confirm the current strategy based on:
            1. The agent's latest state information (targets, sectors, battery constraints, teammate statuses, etc.).
            2. The most recent instructions or intent from the Decision Maker.
            3. The existing overall strategy.

            **GUIDELINES**:
            - Determine whether the current strategy remains valid or requires adjustments
            (e.g., changing priorities, shifting focus to certain sectors, recalibrating objectives).
            - Justify your changes briefly: if you update the strategy, state why.
            - Your output should be concise and actionable, describing only the essential points
            for the agent to follow.

            **OUTPUT FORMAT**:
            "new_strategy": "<short description of the updated or confirmed strategy>"

            **EXAMPLE RESPONSE**:
            "new_strategy": "Focus on securing sectors 3 and 5 while maintaining minimal energy consumption."

            Keep your final strategy statement clear and goal-oriented,
            reflecting both the Decision Maker’s latest message and the current situation.
            """
        )

        human_prompt = """
        Please review the previous state information, the most recent message from the Decision Maker,
        and the current strategy. Then decide whether the existing strategy needs to be adjusted or
        if it remains valid.

        If you decide to revise the strategy, briefly explain (in your own reasoning) why it needs to change.
        OUTPUT:
        "new_strategy": "<updated or confirmed strategy>"
        """

        messages = [
            SystemMessage(system_prompt),
            HumanMessage(human_prompt),
        ]

        # Dynamically create structure output
        model_fields = {"new_strategy": (str, ...)}

        StrategyOutput = self.create_structure_output("StrategyOutput", model_fields)

        # Inference model
        structured_llm = self.llm.with_structured_output(StrategyOutput)
        response = structured_llm.invoke(messages)

        strategy_message = response.new_strategy

        if current_state.get("verbose"):
            print(f"Strategy\nStrategy:{strategy_message}\n" + "-" * 50)

        return {"strategy": strategy_message}
