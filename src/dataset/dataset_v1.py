from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy.engine import Row
from sqlalchemy import select
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar, List
import random
from itertools import chain
from bisect import bisect_left

from src.database.db import (
    AgentTable,
    MessageT,
    Messages,
    MissionProgress,
    PositionT,
    Positions,
    SamplesTable,
    Strategy,
    StrategyT,
)
from src.database.scripts import connect_to_db
from pydantic import BaseModel
from typing import Generic, TypeVar
from typing import Any
from loguru import logger
from datetime import datetime
import json
import tqdm
import pandas as pd

from src.dataset.base.tokens import TokensMapping

T = TypeVar("T")
C = TypeVar("C", bound="BaseConverter")
E = TypeVar("E", bound="BaseExtractor")
D = TypeVar("D")
S = TypeVar("P")
P = TypeVar("S")

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


class BaseConverter(ABC, Generic[D]):
    @abstractmethod
    def convert(self, data_from_db: list[D] | tuple[list[D], list[D]]) -> TimelineData:
        """Convert one sample extracted from DB to the TimelineData."""
        ...


class BaseExtractor(ABC, Generic[D]):
    def __init__(self, session: Session):
        super().__init__()
        self.session = session

    @abstractmethod
    def fetch_data(self, **kwargs) -> list[D] | tuple[list[D], list[D]]:
        ...


class BaseProcessor(ABC, Generic[C, E, S]):
    def __init__(self, converter: C, extractor: E):
        self._converter = converter
        self._extractor = extractor

    def _fetch_all(self, **kwargs):
        return self._extractor.fetch_data(**kwargs)

    def convert_all(self, **kwargs) -> S:
        raws = self._fetch_all(**kwargs)
        return self._converter.convert(raws)

    @abstractmethod
    def process(self, **kwargs) -> list[TimelineData]:
        ...


class MessageConverter(BaseConverter[Messages]):
    """Convert sequence of messages from one sample.

    Args:
        BaseConverter (_type_): _description_

    Returns:
        _type_: _description_
    """

    def convert(self, data_from_db: FetchedMessagesModel) -> TimelineData:
        sent_messages: FetchedMessagesT = data_from_db.sent_messages
        recieved_messages: FetchedMessagesT = data_from_db.recieved_messages

        processed_sent_messages = [
            self._convert_sent_message(message) for message in sent_messages
        ]

        processed_recieved_messages = [
            self._convert_recieved_message(message) for message in recieved_messages
        ]

        return processed_sent_messages + processed_recieved_messages

    def _convert_sent_message(self, row: tuple[Messages]) -> TimelineData:
        row = row[0]

        receiver_agent_no = row.receiver.agent_no

        if row.type == MessageT.INFO:
            message_type = TokensMapping.INFO_TOKEN.value
        elif row.type == MessageT.ORDER:
            message_type = TokensMapping.ORDER_TOKEN.value
        else:
            raise ValueError(f"Unknown message type: {row.type}. ")

        processed_message = self._process_message_with_tokens(row.message)

        parts = [
            TokensMapping.SENT_MESSAGE_TOKEN.value,
            TokensMapping.TO_TOKEN.value,
            TokensMapping.AGENT_ID_FORMAT.get_agent_token(receiver_agent_no),
            message_type,
            TokensMapping.MESSAGE_TOKEN.value,
            processed_message,
        ]

        formatted = " ".join(parts)

        return TimelineData(
            timestamp=row.timestamp,
            formatted=formatted,
            type="sent_message",
        )

    def _convert_recieved_message(self, row: tuple[Messages]) -> TimelineData:
        row = row[0]

        sender_agent_no = row.sender.agent_no

        if row.type == MessageT.INFO:
            message_type = TokensMapping.INFO_TOKEN.value
        elif row.type == MessageT.ORDER:
            message_type = TokensMapping.ORDER_TOKEN.value
        else:
            raise ValueError(f"Unknown message type: {row.type}.")

        processed_message = self._process_message_with_tokens(row.message)

        parts = [
            TokensMapping.RECEIVED_MESSAGE_TOKEN.value,
            TokensMapping.AGENT_ID_FORMAT.get_agent_token(sender_agent_no),
            TokensMapping.TO_ME_TOKEN.value,
            message_type,
            TokensMapping.MESSAGE_TOKEN.value,
            processed_message,
        ]

        formatted = " ".join(parts)

        return TimelineData(
            timestamp=row.timestamp,
            formatted=formatted,
            type="recieved_message",
        )

    def _process_message_with_tokens(self, message: str):
        processed_message = TokensMapping.process_message_with_agent_tokens(message)
        processed_message = TokensMapping.process_message_with_position_tokens(
            processed_message
        )
        return processed_message


class MessageExtractor(BaseExtractor[Messages]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_data(self, sample_id: int, agent_id: int) -> FetchedMessagesModel:
        """Collect all messages from the Message table.

        Args:
            sample_id (int): id of the sample
            agent_id (int): id of the agent in the DB
        """
        from sqlalchemy.orm import selectinload

        sent_messages = self.session.execute(
            select(Messages)
            .options(selectinload(Messages.receiver))
            .where((Messages.sample_id == sample_id) & (Messages.sender_id == agent_id))
        ).fetchall()

        recieved_messages = self.session.execute(
            select(Messages)
            .options(selectinload(Messages.sender))
            .where(
                (Messages.sample_id == sample_id) & (Messages.receiver_id == agent_id)
            )
        ).fetchall()

        return FetchedMessagesModel(
            sent_messages=sent_messages, recieved_messages=recieved_messages
        )


class DefaultMessageProcessor(
    BaseProcessor[MessageConverter, MessageExtractor, TimelineData]
):
    def __init__(self, converter: MessageConverter, extractor: MessageExtractor):
        super().__init__(converter, extractor)

    def process(self, sample_id: int, agent_id: int) -> list[TimelineData]:
        return self.convert_all(sample_id=sample_id, agent_id=agent_id)


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


class BaseSpliter(ABC, Generic[T, S, P]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data_to_be_splitted: S) -> P:
        splitted_data = self.split(data_to_be_splitted)
        return self.post_process(splitted_data)

    @abstractmethod
    def split(self, timeline_data: S) -> T:
        ...

    @abstractmethod
    def post_process(self, splitted_data: T) -> P:
        ...


class SlicingWindowSplitter(
    BaseSpliter[SplittedData, list[TimelineData], list[PostProcessedSample]]
):
    def __init__(
        self,
        target_column: str = "sent_message",
        max_window_size: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_column = target_column
        self.max_window_size = max_window_size

    def post_process(self, splitted_data: SplittedData) -> list[PostProcessedSample]:
        post_processed = []
        for learning_data, target_data in splitted_data:
            post_processed.append(
                PostProcessedSample(
                    learning_data="\n".join(td.formatted for td in learning_data),
                    target_data=target_data.formatted,
                    rollout_length=len(learning_data),
                    start_timestamp=sorted([td.timestamp for td in learning_data])[0],
                    end_timestamp=sorted([td.timestamp for td in learning_data])[-1],
                    target_timestamp=target_data.timestamp,
                )
            )
        return post_processed

    def split(self, timeline_data: list[TimelineData]) -> SplittedData:
        """Split a mixed timeline into training windows and target events.

        For each event in `timeline_data` whose `type` matches `self.target_column`
        (the target event), collect all preceding events (where `type` does not
        match `self.target_column`) whose `timestamp` falls within the interval
        `[target.timestamp - self.max_window_size, target.timestamp)`.
        Training events are first sorted by ascending `timestamp`.
        If no training events fall into that window, the target event is skipped.

        Args:
            timeline_data (list[TimelineData]): A list of `TimelineData` objects containing both training and target events.

        Returns:
            list[tuple[list[TimelineData], TimelineData]]: A list of `(window, target_event)` tuples, where:
            - `window` is a list of training `TimelineData` objects occurring
              before the target event, sorted by ascending `timestamp`.
            - `target_event` is the `TimelineData` object corresponding to the
              target event.
        """
        x_target_samples = []
        target_timeline_data = []
        learning_timeline_data = []

        for td in timeline_data:
            if td.type == self.target_column:
                target_timeline_data.append(td)
            else:
                learning_timeline_data.append(td)

        learning_timeline_data.sort(key=lambda td: td.timestamp)

        all_target_timestamp = [(d.timestamp, d) for d in target_timeline_data]
        all_learning_timestamps = [d.timestamp for d in learning_timeline_data]

        for target_t, target_v in all_target_timestamp:
            lower_bound = target_t - self.max_window_size
            if lower_bound < 0:
                lower_bound = 0

            i = bisect_left(all_learning_timestamps, lower_bound)
            j = bisect_left(all_learning_timestamps, target_t)

            samples = learning_timeline_data[i:j]

            if not samples:
                continue

            x_target_samples.append((samples, target_v))

        return x_target_samples


class MetadataJSON(BaseModel):
    metadata: list[SampleMetadata]


class BaseDataProcessor(ABC):
    """

    Iterating over sample ID.
    - Each sample should be assigned to Train or Test
    - Each sample should have annotations:
        - path: path/to/sample
        - id_in_db: (int)
        - agent_role: (str) scout / rescuer / scout_commander
        - rollout_length: (int)
        - dataset_version: (int)
    - Each sample contains data from N number of agents
    - For each instance (sent_messages) should be prepared separate strategy for sampling
    """

    DATASET_VERSION: int = 0

    def __init__(
        self, data_dir: Path, metadata_path: Path | None, session: Session, **kwargs
    ):
        self.session = session
        self.metadata_path = metadata_path
        self.data_dir = data_dir
        self.processors = self.initialize_processors()
        self.dataset_version = self.DATASET_VERSION
        self.all_processed_data = []

    @abstractmethod
    def initialize_processors(self, **kwargs) -> list[BaseProcessor]:
        ...

    @abstractmethod
    def split(self, data_to_be_splitted, **kwargs):
        ...

    @abstractmethod
    def post_process(self, processed_data) -> list[PostProcessedSample]:
        ...

    @abstractmethod
    def record_sample_and_annotation(
        self,
        post_processed_sample: list[PostProcessedSample],
        sample_id: int,
        agent_id: int,
        sample_path: Path,
        annotation_path: Path,
    ):
        ...

    def get_all_ids(self) -> list[int]:
        stmt = select(SamplesTable.id)
        result = self.session.execute(stmt).scalars().all()
        return result

    def process_all_samples_in_db(self):
        """Process all samples inside the given directory"""

        samples_path, annotations_path = self._prepare_dir()
        logger.info(
            f"Directory for samples and annotations has been prepared: {samples_path}, {annotations_path}."
        )

        all_processed_samples = []
        all_ids = self.get_all_ids()
        logger.info(f"Find {len(all_ids)} samples inside the Database.")

        if len(all_ids) == 0:
            logger.warning("Samples not found inside the Database.")
            return None

        for sample_id in tqdm.tqdm(all_ids):
            post_processed_sample = self.process_sample(sample_id)
            all_processed_samples.append(post_processed_sample)

        logger.info("Start saving process.")

        for idx, (agent, post_processed_sample) in enumerate(
            [
                (agent, sd)
                for agent, all_splitted_data in all_processed_samples
                for sd in all_splitted_data
            ]
        ):

            sample_path = samples_path / f"{idx:04d}.json"
            annotation_path = annotations_path / f"{idx:04d}.json"

            # `agent.sample_id` and `agent.id` are used for annotations only.
            self.record_sample_and_annotation(
                post_processed_sample,
                agent.sample_id,
                agent.id,
                sample_path,
                annotation_path,
            )

        logger.info("All samples has been saved.")

    def process_sample(self, sample_id: int) -> tuple[AgentTable, PostProcessedSample]:
        """Sample ID from db

        1. Get info about agent
        2. For each agent :
            2.1. Call all methods to collect the data from DB
            2.2. Use converter to get the post-processed data
            2.3. Save sample JSON, update Metadata.json
        """
        agents = self._get_info_about_agents(sample_id)

        for agent in agents:
            assert (
                len(agent) == 1
            ), f"`process_agent` requires agent to be tuple, e.g. (AgentTable, ), got {type(agent)}"

            agent = agent[0]

            processed_data = list(
                chain.from_iterable(
                    processor.process(agent.sample_id, agent.id)
                    for processor in self.processors
                )
            )

            post_processed_data = self.post_process(processed_data)
            splitted_data = self.split(post_processed_data)

            return agent, splitted_data

    def _prepare_dir(self) -> tuple[Path, Path]:
        if self.data_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.data_dir = Path(
                f"{self.data_dir}_{timestamp}_dataset_version_{self.dataset_version}"
            )
            self.data_dir.mkdir(parents=True)
            logger.info(
                f"Data dir already exists: {self.data_dir}. Created new one: {self.data_dir}"
            )
        else:
            self.data_dir.mkdir(parents=True)
            logger.info(f"Saving data into {self.data_dir}...")

        samples_path = self.data_dir / "samples"
        annotations_path = self.data_dir / "annotations"

        samples_path.mkdir(parents=True)
        annotations_path.mkdir(parents=True)

        return samples_path, annotations_path

    def _get_info_about_agents(self, sample_id: int = 1):
        stmt = select(AgentTable).where(AgentTable.sample_id == sample_id)
        return self.session.execute(stmt).fetchall()


class AlpacaTemplate(BaseModel):
    instruction: str
    input: str
    output: str
    system: str


class AlpacaDatasetV1(BaseDataProcessor):

    DATASET_VERSION = 1
    TARGET_COLUMN: int = "sent_message"

    def __init__(
        self, data_dir: Path, metadata_path: Path | None, session: Session, **kwargs
    ):
        super().__init__(data_dir, metadata_path, session, **kwargs)

    def initialize_processors(self, **kwargs):
        return [
            DefaultMessageProcessor(
                converter=MessageConverter(),
                extractor=MessageExtractor(session=self.session),
            )
        ]

    def split(self, timeline_data, **kwargs) -> list[PostProcessedSample]:
        timeline_data.sort(key=lambda td: td.timestamp)
        splitter = SlicingWindowSplitter(target_column=self.TARGET_COLUMN, **kwargs)
        return splitter(timeline_data)

    def post_process(self, processed_data: list[TimelineData]) -> list[TimelineData]:
        """Add special time delta token.

        Time delta token is calculated as a diff of previous and the next events.
        For example, for the following two time events
            (1) timestamp = 10
            (2) timestamp = 10
            (3) timestamp = 15
            (3) timestamp = 15

            the tokens will be

            <T+0> ...
            <T+0> ...
            <T+5> ...
            <T+0> ...
        Args:
            processed_data (list[TimelineData]): _description_

        Returns:
            _type_: _description_
        """

        df = pd.DataFrame([td.model_dump() for td in processed_data])
        only_learning_data = df[~(df["type"] == self.TARGET_COLUMN)]
        target_data = df[df["type"] == self.TARGET_COLUMN]
        only_learning_data.sort_values(by="timestamp", inplace=True)
        only_learning_data["time_delta"] = (
            only_learning_data["timestamp"].diff().fillna(0).astype(int)
        )
        only_learning_data["time_delta_token"] = only_learning_data["time_delta"].apply(
            lambda x: TokensMapping.get_time_token(x)
        )
        only_learning_data["formatted"] = (
            only_learning_data["time_delta_token"]
            + " "
            + only_learning_data["formatted"]
        )
        only_learning_data = only_learning_data.drop(
            columns=["time_delta", "time_delta_token"]
        )
        updated_processed_data = [
            TimelineData(**row) for _, row in only_learning_data.iterrows()
        ]
        target_data = [TimelineData(**row) for _, row in target_data.iterrows()]
        return updated_processed_data + target_data

    def record_sample_and_annotation(
        self,
        post_processed_sample: PostProcessedSample,
        sample_id: int,
        agent_id: int,
        sample_path: Path,
        annotation_path: Path,
    ):
        try:
            sample_dict = AlpacaTemplate(
                instruction="",
                input=post_processed_sample.learning_data,
                output=post_processed_sample.target_data,
                system="",
            )

            sample_ann = SampleMetadata(
                id_in_db=sample_id,
                agent_id=agent_id,
                path=str(sample_path),
                dataset_version=self.dataset_version,
                rollout_length=post_processed_sample.rollout_length,
                start_timestamp=post_processed_sample.start_timestamp,
                end_timestamp=post_processed_sample.end_timestamp,
                target_timestamp=post_processed_sample.target_timestamp,
            )

            with open(sample_path, "w", encoding="utf-8") as f:
                json.dump(sample_dict.model_dump(), f, ensure_ascii=False, indent=2)

            with open(annotation_path, "w", encoding="utf-8") as f:
                json.dump(
                    sample_ann.model_dump(),
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            logger.error(
                f"Failed to record sample or annotation for sample_id={sample_id}, agent_id={agent_id}: {e}"
            )


if __name__ == "__main__":
    db_path = Path(".database/data.db")
    annotation_path = None
    session = connect_to_db(db_path)
    dataset = AlpacaDatasetV1(Path("./dataset"), annotation_path, session)
    dataset.process_all_samples_in_db()
