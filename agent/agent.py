import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import Annotated, TypedDict


class NextAction(TypedDict):
    """Action should be taken by the agent"""

    explaination: Annotated[
        str, ..., "Short explanaition why this aciton should be taken"
    ]
    action: Annotated[int, ..., "Action should be taken"]


class Agent:
    def __init__(
        self,
        area_size: tuple[int, int],
        object_map: dict[str, int],
        api_key: str,
        chat_gpt_model: str = "gpt-3.5-turbo",
        observation_area: tuple[int, int] = (9, 9),
    ):
        self.object_map = object_map
        self.area_size = area_size
        self.observation_area = observation_area
        self.model = ChatOpenAI(model=chat_gpt_model, api_key=api_key)
        self._setup_agent()

    def _setup_agent(self):
        self.visited_map = np.full(
            (self.area_size[0], self.area_size[1]),
            self.object_map.get("NOT_VISITED_AREA"),
        )
        self.current_observation = None

    @staticmethod
    def generate_action():
        possible_actions = [0, 1, 2, 3, 4]
        return np.random.choice(possible_actions)

    def generate_action_by_model(self, visited_map, observation):
        system_propmt = self._get_system_prompt()
        prompt = self._get_prompt(visited_map, observation)
        messages = [
            SystemMessage(content=system_propmt),
            HumanMessage(content=prompt),
        ]

        structured_llm = self.model.with_structured_output(NextAction)
        response = structured_llm.invoke(messages)
        return response

    def _get_system_prompt(self) -> str:
        return f"""
        You are controlling an autonomous drone in a grid-based area.
        The grid contains visited and unvisited cells. Your task is to decide the next action.
        The possible actions are:
        0 - Move up
        1 - Move down
        2 - Move left
        3 - Move right
        4 - Stay in place

        The map has the following structure:
        ```
        OBJECT_MAP = {{
            'BACKGROUND': 0,
            'OBSTACLE': 1,
            'TARGET_POINT': 2,
            'NOT_VISITED_AREA': 3,
            'AGENT_POSITION': 4
        }}
        ```
        Remember to omit the obstacles. If you hit an obstacle,
        you will fail the mission.
        Your goal: find the TARGET_POINT if target point will appeat in the observation area - you win!
        Provide your action as a single integer (0, 1, 2, 3, or 4).
        Output examples: Next action - 1
        """

    def _get_prompt(self, visited_map, observation) -> str:
        return f"""
            Here is the current state of the drone:
            Visited map: {visited_map}
            Observation area size: {self.get_observation_area()}
            Current observation: {observation}
        Output examples: Next action - 1
        """

    def update_visited_map(
        self, observation: np.ndarray, binary_observation_mask: np.ndarray
    ) -> None:
        """
        Updates the visited area with the observation in the regions specified by the binary observation mask.

        Parameters:
            visited_area (np.ndarray): The map of visited areas to be updated.
            observation (np.ndarray): The observed values to be placed on the visited area.
            binary_observation_mask (np.ndarray): A binary mask indicating where the observation applies.

        Returns:
            None: Updates are made directly to the visited_area.
        """
        # Find the bounding box of the mask
        rows, cols = np.where(binary_observation_mask == 1)
        if len(rows) == 0 or len(cols) == 0:
            return  # Mask is empty, nothing to update

        min_row, max_row = rows.min(), rows.max() + 1
        min_col, max_col = cols.min(), cols.max() + 1

        # Extract the target region in the mask
        mask_height = max_row - min_row
        mask_width = max_col - min_col

        # Ensure the observation aligns with the mask's region
        if observation.shape != (mask_height, mask_width):
            raise ValueError(
                "Observation shape does not match the size of the binary observation mask region."
            )

        # Update the visited area with the observation values
        self.visited_map[min_row:max_row, min_col:max_col] = observation

    def get_observation_area(self):
        return self.observation_area

    def get_visited_map(self):
        return self.visited_map
