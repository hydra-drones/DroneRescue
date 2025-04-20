import logging
from typing import Literal

logging.basicConfig(level=logging.INFO)


class Agent:
    def __init__(
        self,
        agent_id,
        role,
        sensor_range,
        start_timestamp: int,
        start_position: tuple[int, int],
        mission: str,
        global_strategy: str,
        verbose: bool = True,
    ):
        """
        messages_from_agents: dict of messages from agents, where the key is
            the timestamp and the value is a dict of messages from agents and their IDs
        sended_messages: dict of messages sent by the agent, where the key is
            the timestamp and the value is a dict of messages and IDs of agent recipient
        """
        self.agent_id = agent_id
        self.role = role
        self.position = start_position
        self._mission = mission
        self.verbose = verbose
        self._sensor_range = sensor_range
        self._messages_from_agents: dict[int, dict[int, str]] = {}
        self._sended_messages: dict[int, dict[int, str]] = {}
        self._positions: dict[int, tuple[int, int]] = {start_timestamp: start_position}
        self._mission_progress: dict[int, str] = {}
        self._latest_agent_information: dict[int, dict[int, dict[str, str]]] = {}
        self._actions: dict[int, str] = {}
        self._global_strategy: dict[int, str] = {start_timestamp: global_strategy}
        self._local_strategy: dict[int, str] = {}
        self._special_actions: dict[int, str] = {}
        self._target_in_fov: dict[int, list[tuple[int, int]]] = {}
        self._current_timestamp: int = start_timestamp
        self._setup_context()

    def _setup_context(self):
        """Setup context for the agent"""
        if self.verbose:
            logging.info("Agent %s created at %s", self.agent_id, self.position)

        self._context = {
            "agent_id": self.agent_id,
            "role": self.role,
            "mission": self._mission,
        }

    def update_position_in_edit_mode(self, new_position: tuple[int, int]):
        """Update position in edit mode. Timestamp is not updated

        Set new position under the current timestamp
        """
        self._set_new_position(new_position)
        if self.verbose:
            logging.info(
                "EDIT MODE: Agent %s updated position to %s",
                self.agent_id,
                self.position,
            )

    def update_timestamp_and_set_new_position(
        self, step_in_time: int, new_position: tuple[int, int]
    ):
        """Update timestamp and set new position"""
        self._increase_timestamp(step_in_time)
        self._set_new_position(new_position)

    def get_latest_timestamp(self) -> int:
        """Get the latest timestamp"""
        return self._current_timestamp

    def increase_local_timestamp_to_global_and_sync_position(
        self, global_timestamp: int
    ):
        """Increase local timestamp to global timestamp"""
        if self.verbose:
            logging.info(
                "Agent %s increased local timestamp to global timestamp %s",
                self.agent_id,
                global_timestamp,
            )
        current_position = self._positions[self._current_timestamp]
        self._current_timestamp = global_timestamp
        self._positions[self._current_timestamp] = current_position

    def _increase_timestamp(self, timestamp: int):
        """Update the timestamp"""
        if self.verbose:
            logging.info("Agent %s updated timestamp to %s", self.agent_id, timestamp)
        self._current_timestamp += timestamp
        logging.info("Updated timestamp: %s", self._current_timestamp)

    def _set_new_position(self, new_position: tuple[int, int]):
        """Add position with timestamp"""
        self.position = new_position
        self._positions[self._current_timestamp] = new_position
        if self.verbose:
            logging.info("Agent %s moved to %s", self.agent_id, new_position)

    def get_visiable_targets_in_fov(
        self, target_positions: list[tuple[int, int]]
    ) -> list[tuple[int, int]] | list:
        """Check if target is in field of view radius"""
        fov_radius = self._sensor_range
        targets_in_fov = []

        for target_pos in target_positions:
            # Calculate Euclidean distance between agent and target
            distance = (
                (self.position[0] - target_pos[0]) ** 2
                + (self.position[1] - target_pos[1]) ** 2
            ) ** 0.5

            # Check if target is within field of view radius
            if distance <= fov_radius:
                targets_in_fov.append(target_pos)

        return targets_in_fov

    def add_message_from_agent(
        self,
        global_timestamp: int,
        sender_id: int,
        message: str,
        message_type: Literal["info", "order"],
    ):
        """Add message from agent to the list of messages"""

        self._messages_from_agents[global_timestamp] = {
            "sender_id": sender_id,
            "message": message,
            "type": message_type,
        }

        if self.verbose:
            logging.info(
                "Agent %s received message from agent %s at %s",
                self.agent_id,
                sender_id,
                global_timestamp,
            )

    def add_sended_message(
        self,
        global_timestamp: int,
        receiver_id: int,
        message: str,
        message_type: Literal["info", "order"],
    ):
        """Add sended message with id of receipient agent"""
        if self.verbose:
            logging.info(
                "Agent %s sent message to agent %s: %s",
                self.agent_id,
                receiver_id,
                message,
            )
        self._sended_messages[global_timestamp] = {
            "receiver": receiver_id,
            "message": message,
            "type": message_type,
        }

    def update_information_about_agents(
        self, agents_information: dict[int, dict[str, str]]
    ) -> None:
        """Add information dictionaery about agents to current timestamp
        agents_information - key - agent ids, value: dict with information about agent
        """

        logging.info(
            "Agent %s updated information about agents at %s",
            self.agent_id,
            self._current_timestamp,
        )

        self._latest_agent_information[self._current_timestamp] = agents_information

    def get_current_information_about_agent(self) -> dict[str, str]:
        """Return the information about the agent"""
        return {
            "position": self.position,
            "timestamp": self._current_timestamp,
        }

    @property
    def agent_state(self):
        """Return the state of the agent"""

        action = {}
        special_action = {}

        if self._actions:
            action = self._actions[list(self._actions.keys())[-1]]

        if self._special_actions:
            special_action = self._special_actions[
                list(self._special_actions.keys())[-1]
            ]

        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "position": self.position,
            "timestamp": self._current_timestamp,
            "action": action,
            "special_action": special_action,
        }

    def update_mission_progress(self, progress: str):
        """Update the mission progress"""
        if self.verbose:
            logging.info(
                "Agent %s updated mission progress at %s",
                self.agent_id,
                self._current_timestamp,
            )

        self._mission_progress[self._current_timestamp] = progress

    def update_global_strategy(self, new_global_strategy: str):
        """Update global strategy"""
        if self.verbose:
            logging.info(
                "Agent %s updated global strategy at %s",
                self.agent_id,
                self._current_timestamp,
            )

        self._global_strategy[self._current_timestamp] = new_global_strategy

    def update_local_strategy(self, new_local_strategy: str):
        """Update local strategy"""
        if self.verbose:
            logging.info(
                "Agent %s updated local strategy at %s",
                self.agent_id,
                self._current_timestamp,
            )

        self._local_strategy[self._current_timestamp] = new_local_strategy

    def update_target_in_fov(self, targets: list[tuple[int, int]]):
        """Update target in fov

        Target is a list of tuples with coordinates of the target
        """
        if self.verbose:
            logging.info(
                "Agent %s updated target in fov at %s",
                self.agent_id,
                self._current_timestamp,
            )

        self._target_in_fov[self._current_timestamp] = targets

    def add_action(self, global_timestamp: int, new_x: int, new_y: int):
        """Add action to the list of actions"""
        if self.verbose:
            logging.info(
                "Agent %s decided to go to position %s, %s at %s",
                self.agent_id,
                new_x,
                new_y,
                global_timestamp,
            )
        self._actions[global_timestamp] = (new_x, new_y)

    def add_special_action(self, global_timestamp: int, action: str):
        """Add special action to the list of actions"""
        if self.verbose:
            logging.info(
                "Agent %s decided to %s at %s", self.agent_id, action, global_timestamp
            )

        self._special_actions[global_timestamp] = action

    def freeze_agent_state(self) -> dict:
        """Convert the agent state into dict"""
        return {
            "role": self.role,
            "mission": self._mission,
            "messages_from_agents": self._messages_from_agents,
            "sended_messages": self._sended_messages,
            "positions": self._positions,
            "mission_progress": self._mission_progress,
            "target_in_fov": self._target_in_fov,
            "latest_agents_information": self._latest_agent_information,
            "actions": self._actions,
            "global_strategy": self._global_strategy,
            "local_strategy": self._local_strategy,
            "special_actions": self._special_actions,
        }
