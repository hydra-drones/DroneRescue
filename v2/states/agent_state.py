from typing import List, Tuple, Dict, Union
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Provides the scheme of Agent State"""

    # Messages State
    messages: List[str]

    # Transitions
    next_agent: List[str]

    # About agent
    agent_id: str
    role: str

    # Environment data
    env_size: tuple[int]
    action_history: List[int]
    speed_history: List[int]
    trajectory_history: List[int]

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
    messages_from_agents: Tuple[str, str]
    message_to_teammate_agent: List[tuple]

    # Static Variables
    actions: Tuple[str]
    speeds: Tuple[str]
    observation_area: Tuple[int]
    verbose: bool

    # ChatGPT Configuration
    chatgpt_api_key: str

    def __init__(
        self,
        agent_id: str,
        role: str,
        env_size: tuple[int],
        messages: List[str],
        observation_area: Tuple[int],
        state_description: str,
        action_history: List[int],
        speed_history: List[int],
        trajectory_history: List[tuple],
        next_agent: List[str],
        strategy: str,
        battery_level: float,
        current_position: Tuple[int],
        current_sector: int,
        map_of_sectors: str,
        areas_of_potential_target_locations: List[Tuple[int]],
        observation: List[List[int]],
        teammate_agent_info: List[
            Dict[Union[str, int], Dict[str, Union[str, Tuple[int]]]]
        ],
        messages_from_agents: Tuple[str, str],
        message_to_teammate_agent: List[tuple],
        chatgpt_api_key: str,
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
        self.actions = actions
        self.speeds = speeds
        self.verbose = verbose
