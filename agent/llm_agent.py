from langchain_openai import ChatOpenAI
from agent.base_agent import BaseDroneAgent

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict

from agent.prompts import generate_drone_state_prompt, generate_system_prompt


class NextAction(TypedDict):
    """Action should be taken by the agent"""

    explaination: Annotated[
        str, ..., "Short explanaition why this aciton should be taken"
    ]
    action: Annotated[int, ..., "Action should be taken"]


class LLMAgent(BaseDroneAgent):
    def __init__(
        self,
        area_size: tuple[int, int],
        object_map: dict[str, int],
        api_key: str,
        observation_area: tuple[int, int] = (9, 9),
        chat_gpt_model: str = "gpt-3.5-turbo",
    ):
        super().__init__(
            area_size=area_size,
            object_map=object_map,
            observation_area=observation_area,
        )

        self.model = ChatOpenAI(model=chat_gpt_model, api_key=api_key)

    def generate_action_by_model(self, visited_map, observation):
        system_propmt = generate_system_prompt(self.object_map)
        prompt = generate_drone_state_prompt(
            visited_map, observation, self.observation_area
        )
        messages = [
            SystemMessage(content=system_propmt),
            HumanMessage(content=prompt),
        ]

        structured_llm = self.model.with_structured_output(NextAction)
        response = structured_llm.invoke(messages)
        return response
