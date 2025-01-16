from typing import Dict, Tuple, Any, Literal
from pydantic import create_model
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from v2.states.agent_state import AgentState


class AgentRunnable(Runnable):
    """
    Abstract class. 'invoke' method should be initialized.
    """

    def __init__(
        self, llm_api_key: str, common_agent_state: AgentState, llm_model: str = "gpt-4"
    ):
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.state = common_agent_state
        self.llm = self._initialize_llm()

    @property
    def agent_state(self) -> str:
        """Combine some details about the agent"""
        return f"""
            # System information
            Battery level: {self.state.get("battery_level")}

            # Spatial information
            Current sector: {self.state.get("current_sector")}

            # Communication with teammate agents
            Messages from teammate agents: {self.state.get("messages_from_agents")}
            Teammate agent info: {self.state.get("teammate_agent_info")}
            """

    @property
    def oservation(self) -> str:
        """Combine observation variables"""
        return f"""
            # Position of the agent
            Position: {self.state.get("current_position")}

            # Observation:
            Current observation: {self.state.get("observation")}

            # Observation in coordinates
            {self.state.get("observation_in_coordinates")}
            """

    @property
    def strategy(self) -> str:
        """Combine strategy state"""
        return f"""
        # Current strategy
        Strategy: {self.state.get("strategy")}
        """

    def _initialize_llm(self):
        try:
            llm = ChatOpenAI(
                model=self.llm_model, temperature=0, openai_api_key=self.llm_api_key
            )
            print("ChatGPT initialized")
            return llm
        except Exception as e:
            raise ValueError(f"Failed to initialize ChatGPT: {e}") from e

    def create_structure_output(self, model_name: str, base_model_dict):
        """Dynamically creates base model for structure llm output"""
        return create_model(model_name, **base_model_dict)
