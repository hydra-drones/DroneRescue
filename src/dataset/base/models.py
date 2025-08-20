from pydantic import BaseModel
from typing import TypeVar, Literal
from sqlalchemy.engine import Row
from src.database.db import Messages, Positions, Strategy, MissionProgress

# Type variables
T = TypeVar("T")
C = TypeVar("C")
E = TypeVar("E")
D = TypeVar("D")
S = TypeVar("P")
P = TypeVar("S")

# Type aliases
FetchedMessagesT = list[Row[tuple[Messages]]]
FetchedPositionsT = list[Row[tuple[Positions]]]
FetchedStrategyT = list[Row[tuple[Strategy]]]
FetchedMissionProgressT = list[Row[tuple[MissionProgress]]]
TimestampT = int
SplittedData = list[tuple[list["TimelineData"], "TimelineData"]]


class FetchedMessagesModel:
    def __init__(
        self, sent_messages: FetchedMessagesT, recieved_messages: FetchedMessagesT
    ):
        self.sent_messages = sent_messages
        self.recieved_messages = recieved_messages


class FetchedPositionsModel:
    def __init__(self, ego_pos: FetchedPositionsT, target_pos: FetchedPositionsT):
        self.ego_pos = ego_pos
        self.target_pos = target_pos


class FetchedStrategyModel:
    def __init__(
        self, local_strategy: FetchedStrategyT, global_strategy: FetchedStrategyT
    ):
        self.local_strategy = local_strategy
        self.global_strategy = global_strategy


class FetchedMisionProgressModel:
    def __init__(self, mission_progress: FetchedMissionProgressT):
        self.mission_progress = mission_progress


class TimelineData(BaseModel):
    """Contains post-processed information for certain timestamp."""

    timestamp: int
    formatted: str
    type: Literal[
        "sent_message", "recieved_message", "position", "strategy", "mission_progress"
    ]


class SampleMetadata(BaseModel):
    id_in_db: int
    agent_id: int
    path: str
    dataset_version: int | str
    rollout_length: int
    start_timestamp: int
    end_timestamp: int
    target_timestamp: int


class PostProcessedSample(BaseModel):
    learning_data: str
    target_data: str
    rollout_length: int
    start_timestamp: int
    end_timestamp: int
    target_timestamp: int

    def convert_metadata_to_dict(self) -> dict:
        return dict


class MetadataJSON(BaseModel):
    metadata: list[SampleMetadata]
