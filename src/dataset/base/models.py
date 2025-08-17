from pydantic import BaseModel
from typing import Generic, TypeVar, Literal
from sqlalchemy.engine import Row
from src.database.db import Messages

# Type variables
T = TypeVar("T")
C = TypeVar("C")
E = TypeVar("E")
D = TypeVar("D")
S = TypeVar("P")
P = TypeVar("S")

# Type aliases
FetchedMessagesT = list[Row[tuple[Messages]]]
TimestampT = int
SplittedData = list[tuple[list["TimelineData"], "TimelineData"]]


class FetchedMessagesModel:
    def __init__(
        self, sent_messages: FetchedMessagesT, recieved_messages: FetchedMessagesT
    ):
        self.sent_messages = sent_messages
        self.recieved_messages = recieved_messages


class TimelineData(BaseModel):
    """Contains post-processed information for certain timestamp."""

    timestamp: int
    formatted: str
    type: Literal["sent_message", "recieved_message"]


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
