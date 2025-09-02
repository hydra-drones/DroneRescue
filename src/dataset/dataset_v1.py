from pathlib import Path
import json
import pandas as pd
from loguru import logger

from src.database.scripts import connect_to_db
from src.dataset.base.processors import BaseDataProcessor
from src.dataset.base.models import TimelineData, PostProcessedSample, SampleMetadata
from src.dataset.message.converter import MessageConverter
from src.dataset.message.extractor import MessageExtractor
from src.dataset.positions.converter import PositionConverter
from src.dataset.positions.extractor import PositionExtractor
from src.dataset.strategy.converter import StrategyConverter
from src.dataset.strategy.extractor import StrategyExtractor
from src.dataset.mission_progress.converter import MissionProgressConverter
from src.dataset.mission_progress.extractor import MissionProgressExtractor
from src.dataset.message.processor import DefaultMessageProcessor
from src.dataset.positions.processor import DefaultPositionProcessor
from src.dataset.strategy.processor import DefaultStrategyProcessor
from src.dataset.mission_progress.processor import DefaultMissionProgressProcessor
from src.dataset.metadata.converter import MetadataConverter
from src.dataset.metadata.extractor import MetadataExtractor
from src.dataset.metadata.processor import DefaultMetadataProcessor
from src.dataset.splitters.slicing_window import SlicingWindowSplitter
from src.dataset.templates.alpaca import AlpacaTemplate
from src.dataset.base.tokens import TokensMapping


class AlpacaDatasetV1(BaseDataProcessor):
    """
    Version 1 implementation of the Alpaca dataset processor for drone rescue missions.

    This processor transforms raw drone rescue simulation data into Alpaca-format
    training data suitable for large language models. It handles the complete pipeline
    from database extraction to formatted dataset creation, focusing on message
    communication between autonomous agents during rescue operations.

    Key features:
        - Processes message communication data between rescue agents
        - Adds temporal context through time delta tokens
        - Generates Alpaca-format (instruction, input, output) training samples
        - Supports sliding window data splitting for sequence learning
        - Creates comprehensive metadata for each training sample

    The processor targets "sent_message" events as learning objectives while using
    all other timeline events as contextual input for training.

    :inherits: :class:`~src.dataset.base.processors.BaseDataProcessor`

    Class Attributes:
        DATASET_VERSION (int): Version identifier for dataset format compatibility
        TARGET_COLUMN (str): Column name identifying target events ("sent_message")

    Example:
        .. code-block:: python

            from pathlib import Path
            from src.database.scripts import connect_to_db

            # Setup database connection
            session = connect_to_db(Path("data.db"))

            # Create dataset processor
            dataset = AlpacaDatasetV1(
                data_dir=Path("./output_dataset"),
                metadata_path=None,
                session=session
            )

            # Process all samples in database
            dataset.process_all_samples_in_db()

    Dataset Output Format:
        Each sample generates two files:
            - **Sample file**: Alpaca-format JSON with instruction/input/output
            - **Annotation file**: Metadata with timestamps, agent info, dataset version

    .. seealso::
        :class:`~src.dataset.base.processors.BaseDataProcessor`
            Base processor with common dataset operations
        :class:`~src.dataset.message.processor.DefaultMessageProcessor`
            Message processing component
        :class:`~src.dataset.templates.alpaca.AlpacaTemplate`
            Alpaca format template structure
    """

    DATASET_VERSION = 1
    TARGET_COLUMN: str = "sent_message"

    def __init__(self, data_dir: Path, metadata_path: Path | None, session, **kwargs):
        """
        Initialize the Alpaca dataset processor.

        :param data_dir: Directory path for storing generated dataset files
        :type data_dir: Path
        :param metadata_path: Optional path for metadata storage (currently unused)
        :type metadata_path: Path | None
        :param session: SQLAlchemy database session for data extraction
        :type session: sqlalchemy.orm.Session
        :param kwargs: Additional configuration parameters
        :type kwargs: dict

        Example:
            .. code-block:: python

                dataset = AlpacaDatasetV1(
                    data_dir=Path("./training_data"),
                    metadata_path=None,
                    session=db_session,
                    window_size=10  # Additional parameter
                )
        """
        super().__init__(data_dir, metadata_path, session, **kwargs)

    def initialize_processors(self, **kwargs):
        """
        Initialize the message processing pipeline.

        Creates and configures the default message processor with converter
        and extractor components for handling drone rescue communication data.

        :param kwargs: Additional configuration parameters for processors
        :type kwargs: dict

        :returns: List containing configured message processor
        :rtype: list[DefaultMessageProcessor]

        Example:
            .. code-block:: python

                processors = dataset.initialize_processors()
                print(f"Initialized {len(processors)} processors")
                # Output: Initialized 1 processors
        """
        return [
            DefaultMessageProcessor(
                converter=MessageConverter(),
                extractor=MessageExtractor(session=self.session),
            ),
            DefaultPositionProcessor(
                converter=PositionConverter(),
                extractor=PositionExtractor(session=self.session),
            ),
            DefaultStrategyProcessor(
                converter=StrategyConverter(),
                extractor=StrategyExtractor(session=self.session),
            ),
            DefaultMissionProgressProcessor(
                converter=MissionProgressConverter(),
                extractor=MissionProgressExtractor(session=self.session),
            ),
            DefaultMetadataProcessor(
                converter=MetadataConverter(),
                extractor=MetadataExtractor(session=self.session),
            ),
        ]

    def split(self, timeline_data, **kwargs) -> list[PostProcessedSample]:
        """
        Split timeline data into training samples using sliding window approach.

        Sorts timeline data chronologically and applies sliding window splitting
        to create overlapping sequences suitable for sequential learning. Each
        window contains contextual events leading up to a target message event.

        :param timeline_data: Chronological timeline data to split
        :type timeline_data: list[TimelineData]
        :param kwargs: Configuration parameters for the splitter
        :type kwargs: dict

        :returns: List of processed samples with learning and target data
        :rtype: list[PostProcessedSample]

        Window Parameters (via kwargs):
            - **window_size**: Number of events per window
            - **step_size**: Step between consecutive windows
            - **min_events**: Minimum events required for valid sample

        Example:
            .. code-block:: python

                samples = dataset.split(
                    timeline_data=timeline_events,
                    window_size=10,
                    step_size=5
                )

                for sample in samples:
                    print(f"Learning events: {len(sample.learning_data)}")
                    print(f"Target: {sample.target_data}")

        .. note::
            Timeline data is automatically sorted by timestamp before splitting
            to ensure chronological order in training sequences.
        """
        timeline_data.sort(key=lambda td: td.timestamp)
        splitter = SlicingWindowSplitter(target_column=self.TARGET_COLUMN, **kwargs)
        return splitter(timeline_data)

    def post_process(self, processed_data: list[TimelineData]) -> list[TimelineData]:
        """
        Add temporal context through time delta tokens to timeline data.

        Enhances timeline data by calculating time differences between consecutive
        events and prepending appropriate time delta tokens to the formatted content.
        This provides temporal context for the language model to understand event
        timing relationships during rescue operations.

        Time delta tokens follow the format ``<T+{seconds}>`` where seconds represents
        the time elapsed since the previous event. Special handling ensures target
        events (sent_message) maintain their original formatting.

        :param processed_data: List of timeline data entries to enhance
        :type processed_data: list[TimelineData]

        :returns: Enhanced timeline data with time delta tokens added
        :rtype: list[TimelineData]

        Token Format:
            - ``<T+0>``: No time elapsed (simultaneous events)
            - ``<T+5>``: 5 seconds elapsed since previous event
            - ``<T+60>``: 60 seconds elapsed since previous event

        Processing Steps:
            1. Separate learning data from target events
            2. Sort learning data chronologically by timestamp
            3. Calculate time differences between consecutive events
            4. Map time deltas to appropriate token representations
            5. Prepend time tokens to formatted content
            6. Recombine learning and target data

        Example:
            .. code-block:: python

                # Input timeline data
                timeline_data = [
                    TimelineData(timestamp=1000, type="position", formatted="<POS> 10 20"),
                    TimelineData(timestamp=1005, type="received_message", formatted="<RCV> Roger"),
                    TimelineData(timestamp=1010, type="sent_message", formatted="<SND> Deploy")
                ]

                # After post-processing
                enhanced_data = dataset.post_process(timeline_data)

                # Results in:
                # <T+0> <POS> 10 20      (first event, no delta)
                # <T+5> <RCV> Roger      (5 seconds after previous)
                # <SND> Deploy           (target event, unchanged)

        .. note::
            Target events (sent_message type) are not modified with time tokens
            as they represent the learning objective rather than input context.

        .. warning::
            This method modifies pandas DataFrames in place. The ``.copy()`` method
            is used to avoid SettingWithCopyWarning when updating DataFrame columns.
        """

        df = pd.DataFrame([td.model_dump() for td in processed_data])
        only_learning_data = df[~(df["type"] == self.TARGET_COLUMN)].copy()
        target_data = df[df["type"] == self.TARGET_COLUMN].copy()
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
        """
        Generate and save Alpaca-format training sample with comprehensive metadata.

        Creates a training sample in Alpaca format (instruction, input, output) and
        generates detailed annotation metadata for tracking sample provenance and
        characteristics. Both files are saved as JSON with UTF-8 encoding.

        :param post_processed_sample: Processed sample containing learning and target data
        :type post_processed_sample: PostProcessedSample
        :param sample_id: Database identifier for the original simulation sample
        :type sample_id: int
        :param agent_id: Database identifier for the agent this sample represents
        :type agent_id: int
        :param sample_path: File path where the training sample will be saved
        :type sample_path: Path
        :param annotation_path: File path where the annotation metadata will be saved
        :type annotation_path: Path

        :raises Exception: If file writing fails or data serialization errors occur

        Generated Files:
            **Sample File** (Alpaca format):
                - **instruction**: Currently empty string (for future task descriptions)
                - **input**: Concatenated learning data (contextual events)
                - **output**: Target data (expected agent response)
                - **system**: Currently empty string (for future system prompts)

            **Annotation File** (metadata):
                - **id_in_db**: Original database sample identifier
                - **agent_id**: Agent identifier for multi-agent scenarios
                - **path**: File path to the corresponding sample
                - **dataset_version**: Version number for compatibility tracking
                - **rollout_length**: Number of events in the sample sequence
                - **timestamps**: Start, end, and target event timestamps

        Example:
            .. code-block:: python

                # Sample file content (sample.json):
                {
                    "instruction": "",
                    "input": "<T+0> <POS> 10 20 <T+5> <RCV> <AGENT#2> Status report",
                    "output": "<SND> <TO> <AGENT#2> All clear, proceeding",
                    "system": ""
                }

                # Annotation file content (annotation.json):
                {
                    "id_in_db": 42,
                    "agent_id": 3,
                    "path": "/dataset/samples/0042.json",
                    "dataset_version": 1,
                    "rollout_length": 15,
                    "start_timestamp": 1000,
                    "end_timestamp": 1150,
                    "target_timestamp": 1100
                }

        Error Handling:
            Failed operations are logged with detailed error information including
            sample_id and agent_id for debugging purposes. The method continues
            execution rather than raising exceptions to allow batch processing.

        .. seealso::
            :class:`~src.dataset.templates.alpaca.AlpacaTemplate`
                Alpaca format structure definition
            :class:`~src.dataset.base.models.SampleMetadata`
                Annotation metadata structure
        """
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
