from pydantic import BaseModel, model_validator, Field
from typing import Literal
from src.database.db import MessageT, PositionT


class Message(BaseModel):
    """Used for communication between agents."""

    type: MessageT
    message: str


class MessageFromAgent(Message):
    """Message recevied from another agent."""

    sender_id: int


class SentMessage(Message):
    """Sent message to another agent."""

    receiver_id: int = Field(..., alias="receiver")


class Position(BaseModel):
    """Describes position of the object in local coordinates.

    type: the type of the object
    pos_x: Position along the X-axis
    pos_y: Position along the Y-axis

    Note: if the given position has the type list[int, int].
    It can be converted to the correct format with `_coerce_from_list` function

    """

    type: PositionT
    pos_x: int
    pos_y: int

    @classmethod
    def _coerce_from_list(cls, obj, force_type: PositionT | None = None):
        if isinstance(obj, list) and len(obj) == 2:
            return {
                "type": force_type or PositionT.AGENT,
                "pos_x": obj[0],
                "pos_y": obj[1],
            }

        else:
            raise RuntimeError(
                f"Got unexpected type of the Position: {type(obj)}, {obj}. Expected `list[int, int]`"
            )


class AgentPosition(Position):
    """Agent Position in the world coordinates.

    Inherits from :class:`Position` class. Converts the
    list into correct format before validation.

    type: flag that describes that this is position of Agent.

    """

    type: Literal[PositionT.AGENT]

    @model_validator(mode="before")
    @classmethod
    def coerce_from_list(cls, v):
        """Takes :class:`Position` and converts field to the correct format"""
        return cls._coerce_from_list(v, force_type=PositionT.AGENT)


class TargetPosition(Position):
    """Target Position in the world coordinates.

    Inherits from :class:`Position` class. Converts the
    list into correct format before validation.

    type: flag that describes that this is position of Target.

    """

    type: Literal[PositionT.TARGET]

    @model_validator(mode="before")
    @classmethod
    def coerce_from_list(cls, v):
        """Takes :class:`Position` and converts field to the correct format."""
        return cls._coerce_from_list(v, force_type=PositionT.TARGET)


class HomeBasePosition(Position):
    """Base Position in the world coordinates.

    Inherits from :class:`Position` class. Converts the
    list into correct format before validation.

    type: flag that describes that this is position of Base.

    """

    type: Literal[PositionT.BASE]

    @model_validator(mode="before")
    @classmethod
    def coerce_from_list(cls, v):
        """Takes :class:`Position` and converts field to the correct format."""
        return cls._coerce_from_list(v, force_type=PositionT.BASE)


class AgentInformation(BaseModel):
    """Agent information."""

    position: AgentPosition
    timestamp: int


MissionProgressT = dict[int, str]
GlobalStrategyT = dict[str, str]
LocalStrategyT = dict[str, str]
AgentPositionT = dict[int, AgentPosition]
TargetInFovT = dict[int, list[TargetPosition]]
SentMessagesT = dict[int, list[SentMessage]]


class Agent(BaseModel):
    """Describes Agent data scheme."""

    role: str
    """Type of the agent."""

    mission: str
    """Mission of the agent."""

    messages_from_agents: dict[int, list[MessageFromAgent]]
    """Messages grouped by source agent ID."""

    sent_messages: SentMessagesT = Field(..., alias="sended_messages")
    """Sent messages to other agents, grouped by recipient ID."""

    positions: AgentPositionT
    """Agent's positions by tick or other grouping key."""

    target_in_fov: TargetInFovT
    """Target positions, grouped by tick or assignment."""

    latest_agents_information: dict[str, dict[str, AgentInformation]]
    """Latest information about the agent"""

    mission_progress: MissionProgressT
    """Progress of the agent's mission, grouped by tick or phase."""

    global_strategy: GlobalStrategyT
    """Global strategy, grouped by tick or stage."""

    local_strategy: LocalStrategyT
    """Local strategy, grouped by tick or zone."""

    actions: dict
    """Actions that agent takes, grouped by tick. For now, this is empty dict."""

    special_actions: dict
    """Special actions, grouped by tick or category. For now, this is empty dict."""


class Target(BaseModel):
    """Describe all exist targets in the current simulation."""

    position: TargetPosition


class HomeBase(BaseModel):
    """Describe the Base in the current simulation."""

    position: HomeBasePosition


class JSONSampleModel(BaseModel):
    """Base class for the JSON sample from annotation App."""

    agents: dict[str, Agent]
    targets: dict[str, Target]
    bases: dict[str, HomeBase]
    area: list[int, int]


if __name__ == "__main__":
    import json

    test_json_path = "datasamples/0001.json"

    with open(test_json_path, "r") as file:
        sample = json.load(file)

    model = JSONSampleModel.model_validate(sample)

    print(model)
