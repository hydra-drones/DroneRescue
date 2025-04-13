import logging
from typing import Literal

logging.basicConfig(level=logging.INFO)


class Agent:
    def __init__(
        self,
        agent_id,
        role,
        start_timestamp: int,
        start_position: tuple[int, int],
        mission: str,
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
        self._messages_from_agents: dict[int, dict[int, str]] = {}
        self._sended_messages: dict[int, dict[int, str]] = {}
        self._positions: dict[int, tuple[int, int]] = {start_timestamp: start_position}
        self._mission_progress: dict[int, str] = {}
        self._latest_agent_information: dict[int, dict[str, str]] = {}
        self._actions: dict[int, str] = {}
        self._strategy: dict[int, str] = {}
        self._special_actions: dict[int, str] = {}
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

    def add_latest_information_about_agent(
        self, global_timestamp: int, agent_id: int, position: str
    ):
        """Add latest information about agent"""
        if self.verbose:
            logging.info(
                "Agent %s received information from agent %s at %s",
                self.agent_id,
                agent_id,
                global_timestamp,
            )
        self._latest_agent_information[agent_id] = {
            "timestamp": global_timestamp,
            "position": position,
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

    def update_mission_progress(self, global_timestamp: int, progress: str):
        """Update the mission progress"""
        if self.verbose:
            logging.info(
                "Agent %s updated mission progress at %s",
                self.agent_id,
                global_timestamp,
            )

        self._mission_progress[global_timestamp] = progress

    def update_current_strategy(self, global_timestamp: int, strategy: str):
        """Update the mission progress"""
        if self.verbose:
            logging.info(
                "Agent %s updated mission progress at %s",
                self.agent_id,
                global_timestamp,
            )

        self._strategy[global_timestamp] = strategy

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
            "latest_agents_information": self._latest_agent_information,
            "actions": self._actions,
            "strategy": self._strategy,
            "special_actions": self._special_actions,
        }
