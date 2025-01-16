import operator
from typing import Type, List, Tuple, Dict, Sequence, Literal, Annotated, Union
from typing_extensions import TypedDict
from langchain.schema import BaseMessage


class AgentState(TypedDict):
    """Provides the scheme of Agent State"""

    # Messages State
    messages: Annotated[Sequence[BaseMessage], operator.add]

    # Transitions
    next_agent: List[str]

    # About agent
    agent_id: str
    role: str

    # Environment data
    env_size: tuple[int]
    action_history: Annotated[Sequence[BaseMessage], operator.add]
    speed_history: Annotated[Sequence[BaseMessage], operator.add]
    trajectory_history: Annotated[Sequence[Tuple], operator.add]

    # Strategy
    strategy: str
    state_description: str

    # System information
    battery_level: float

    # Spatial information
    current_position: Tuple[int]
    current_sector: int
    areas_of_potential_target_locations: List[Tuple[int]]

    # Observation
    observation: List[float]

    # Communication
    teammate_agent_info: List[Dict[Union[str, int], Dict[str, Union[str, Tuple[int]]]]]
    messages_from_agents: List[Dict[Union[str, int], Dict[str, Union[str, Tuple[int]]]]]
    message_to_teammate_agent: Annotated[Sequence[Dict], operator.add]

    # Static Variables
    actions: Tuple[str]
    speeds: Tuple[str]
    observation_area: Tuple[int]
    verbose: bool

    # ChatGPT Configuration
    chatgpt_api_key: str
    chatgpt_model: str

    def __init__(
        self,
        agent_id: str,
        role: str,
        env_size: tuple[int],
        messages: Annotated[Sequence[BaseMessage], operator.add],
        observation_area: Tuple[int],
        state_description: Annotated[Sequence[BaseMessage], operator.add],
        action_history: Annotated[Sequence[BaseMessage], operator.add],
        speed_history: Annotated[Sequence[BaseMessage], operator.add],
        trajectory_history: Annotated[Sequence[Tuple], operator.add],
        next_agent: List[str],
        strategy: str,
        battery_level: float,
        current_position: Tuple[int],
        current_sector: int,
        map_of_sectors: List[List[int]],
        areas_of_potential_target_locations: List[Tuple[int]],
        observation: List[List[int]],
        teammate_agent_info: List[
            Dict[Union[str, int], Dict[str, Union[str, Tuple[int]]]]
        ],
        messages_from_agents: List[
            Dict[Union[str, int], Dict[str, Union[str, Tuple[int]]]]
        ],
        message_to_teammate_agent: Annotated[Sequence[Dict], operator.add],
        chatgpt_api_key: str,
        chatgpt_model: str = "gpt-3.5-turbo-0125",
        actions: Tuple[str] = ("0", "1", "2", "3", "4"),
        speeds: Tuple[str] = ("0", "1", "2", "3", "4", "5", "6"),
        verbose: bool = False,
    ):
        self.next_agent = next_agent
        self.messages = messages
        self.id = agent_id
        self.role = role
        self.env_size = env_size
        self.observation_area = observation_area
        self.action_history = action_history
        self.speed_history = speed_history
        self.battery_level = battery_level
        self.current_position = current_position
        self.current_sector = current_sector
        self.map_of_sectors = map_of_sectors
        self.trajectory_history = trajectory_history
        self.strategy = strategy
        self.observation = observation
        self.state_description = state_description
        self.areas_of_potential_target_locations = areas_of_potential_target_locations
        self.teammate_agent_info = teammate_agent_info
        self.messages_from_agents = messages_from_agents
        self.message_to_teammate_agent = message_to_teammate_agent
        self.chatgpt_api_key = chatgpt_api_key
        self.chatgpt_model = chatgpt_model
        self.actions = actions
        self.speeds = speeds
        self.verbose = verbose
