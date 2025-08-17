from abc import ABC, abstractmethod
from typing import Generic
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import select
from datetime import datetime
from loguru import logger
import tqdm
from itertools import chain

from src.dataset.base.models import C, E, S, TimelineData, PostProcessedSample
from src.database.db import SamplesTable, AgentTable


class BaseProcessor(ABC, Generic[C, E, S]):
    """
    Abstract base class for processing pipeline components.

    This class orchestrates the data processing pipeline by combining an extractor
    and converter to transform raw database data into structured timeline data.
    It provides a standardized interface for fetching data, converting it, and
    processing it according to specific requirements.

    The processor follows a two-step pattern:
        1. **Extract**: Use the extractor to fetch raw data from database
        2. **Convert**: Use the converter to transform raw data into timeline format

    :param C: Type parameter for the converter component
    :type C: TypeVar, bound to BaseConverter
    :param E: Type parameter for the extractor component
    :type E: TypeVar, bound to BaseExtractor
    :param S: Type parameter for the output data structure
    :type S: TypeVar

    :param converter: Component responsible for data transformation
    :type converter: C
    :param extractor: Component responsible for data extraction
    :type extractor: E

    .. note::
        All concrete processor implementations must inherit from this class and
        implement the :meth:`process` method with specific processing logic.

    Example:
        .. code-block:: python

            class MessageProcessor(BaseProcessor[MessageConverter, MessageExtractor, TimelineData]):
                def process(self, sample_id, agent_id):
                    return self.convert_all(sample_id=sample_id, agent_id=agent_id)

    .. seealso::
        :class:`~src.dataset.message.processor.DefaultMessageProcessor`
            Concrete implementation for message processing
    """

    def __init__(self, converter: C, extractor: E):
        """
        Initialize the processor with converter and extractor components.

        :param converter: Data conversion component
        :type converter: C
        :param extractor: Data extraction component
        :type extractor: E

        Example:
            .. code-block:: python

                processor = MessageProcessor(
                    converter=MessageConverter(),
                    extractor=MessageExtractor(session=db_session)
                )
        """
        self._converter = converter
        self._extractor = extractor

    def _fetch_all(self, **kwargs):
        """
        Fetch raw data using the configured extractor.

        :param kwargs: Parameters passed to the extractor's fetch_data method
        :type kwargs: dict

        :returns: Raw data from the database
        :rtype: Any

        .. note::
            This is an internal method that delegates to the extractor component.
        """
        return self._extractor.fetch_data(**kwargs)

    def convert_all(self, **kwargs) -> S:
        """
        Execute the complete extract-convert pipeline.

        Fetches raw data using the extractor and then converts it using the
        converter to produce structured timeline data.

        :param kwargs: Parameters passed to the extraction step
        :type kwargs: dict

        :returns: Converted data in the target format
        :rtype: S

        :raises ValueError: If extraction or conversion parameters are invalid
        :raises DatabaseError: If data extraction fails

        Example:
            .. code-block:: python

                timeline_data = processor.convert_all(sample_id=1, agent_id=5)
        """
        raws = self._fetch_all(**kwargs)
        return self._converter.convert(raws)

    @abstractmethod
    def process(self, **kwargs) -> list[TimelineData]:
        """
        Process data according to specific requirements.

        This method should implement the specific processing logic for the
        data type. It typically calls :meth:`convert_all` but may include
        additional processing steps such as filtering, sorting, or validation.

        :param kwargs: Processing parameters specific to the implementation
        :type kwargs: dict

        :returns: List of processed timeline data entries
        :rtype: list[TimelineData]

        :raises NotImplementedError: If the method is not implemented in subclass

        Example:
            .. code-block:: python

                processor = MessageProcessor(converter, extractor)
                timeline_data = processor.process(sample_id=1, agent_id=5)
        """
        pass


class BaseDataProcessor(ABC):
    """
    Abstract base class for complete dataset processing workflows.

    This class orchestrates the entire dataset creation pipeline from database
    extraction to final output generation. It manages the processing of multiple
    samples, agents, and data types to create training datasets for machine learning.

    The processing workflow includes:
        1. **Sample Discovery**: Find all samples in the database
        2. **Agent Processing**: Process each agent within each sample
        3. **Data Pipeline**: Extract, convert, and post-process timeline data
        4. **Data Splitting**: Create training windows and target sequences
        5. **Output Generation**: Save processed data and metadata

    Key responsibilities:
        - Database session management
        - Directory structure creation
        - Progress tracking and logging
        - Error handling and recovery
        - Metadata generation

    :param data_dir: Directory where processed datasets will be saved
    :type data_dir: Path
    :param metadata_path: Optional path for additional metadata (currently unused)
    :type metadata_path: Path | None
    :param session: SQLAlchemy database session for data access
    :type session: Session

    .. note::
        Subclasses must implement all abstract methods to define specific
        processing behavior for different dataset formats.

    Example:
        .. code-block:: python

            class AlpacaDatasetV1(BaseDataProcessor):
                DATASET_VERSION = 1

                def initialize_processors(self):
                    return [MessageProcessor(...)]

                def post_process(self, data):
                    # Add time tokens, etc.
                    return processed_data

    .. seealso::
        :class:`~src.dataset.dataset_v1.AlpacaDatasetV1`
            Concrete implementation for Alpaca-format datasets
    """

    DATASET_VERSION: int = 0

    def __init__(
        self, data_dir: Path, metadata_path: Path | None, session: Session, **kwargs
    ):
        """
        Initialize the data processor with configuration and database session.

        :param data_dir: Base directory for saving processed datasets
        :type data_dir: Path
        :param metadata_path: Optional path for metadata files (reserved for future use)
        :type metadata_path: Path | None
        :param session: Active SQLAlchemy database session
        :type session: Session
        :param kwargs: Additional configuration parameters for subclasses
        :type kwargs: dict

        Example:
            .. code-block:: python

                processor = AlpacaDatasetV1(
                    data_dir=Path("./datasets"),
                    metadata_path=None,
                    session=db_session
                )
        """
        self.session = session
        self.metadata_path = metadata_path
        self.data_dir = data_dir
        self.processors = self.initialize_processors()
        self.dataset_version = self.DATASET_VERSION
        self.all_processed_data = []

    @abstractmethod
    def initialize_processors(self, **kwargs) -> list[BaseProcessor]:
        """
        Initialize the data processing pipeline components.

        Create and configure all processor instances needed for the dataset
        creation pipeline. Each processor handles a specific data type
        (e.g., messages, positions, strategies).

        :param kwargs: Configuration parameters for processor initialization
        :type kwargs: dict

        :returns: List of configured processor instances
        :rtype: list[BaseProcessor]

        :raises NotImplementedError: If not implemented in subclass

        Example:
            .. code-block:: python

                def initialize_processors(self):
                    return [
                        MessageProcessor(MessageConverter(), MessageExtractor(self.session)),
                        PositionProcessor(PositionConverter(), PositionExtractor(self.session))
                    ]
        """
        pass

    @abstractmethod
    def split(self, data_to_be_splitted, **kwargs):
        """
        Split timeline data into training sequences and targets.

        Transform processed timeline data into training examples by creating
        input-output pairs suitable for machine learning. This typically
        involves windowing strategies and sequence generation.

        :param data_to_be_splitted: Processed timeline data to split
        :type data_to_be_splitted: Any
        :param kwargs: Splitting configuration parameters
        :type kwargs: dict

        :returns: Split data ready for training
        :rtype: Any

        :raises NotImplementedError: If not implemented in subclass

        Example:
            .. code-block:: python

                def split(self, timeline_data):
                    splitter = SlicingWindowSplitter(max_window_size=100)
                    return splitter(timeline_data)
        """
        pass

    @abstractmethod
    def post_process(self, processed_data) -> list[PostProcessedSample]:
        """
        Apply final processing steps to timeline data.

        Perform any additional transformations needed before data splitting,
        such as adding time delta tokens, normalization, or filtering.

        :param processed_data: Raw timeline data from processors
        :type processed_data: Any

        :returns: Post-processed timeline data ready for splitting
        :rtype: list[PostProcessedSample]

        :raises NotImplementedError: If not implemented in subclass

        Example:
            .. code-block:: python

                def post_process(self, data):
                    # Add time delta tokens
                    # Sort by timestamp
                    # Apply filtering
                    return processed_data
        """
        pass

    @abstractmethod
    def record_sample_and_annotation(
        self,
        post_processed_sample: list[PostProcessedSample],
        sample_id: int,
        agent_id: int,
        sample_path: Path,
        annotation_path: Path,
    ):
        """
        Save processed sample and its metadata to files.

        Write the final processed data and associated metadata to the
        filesystem in the appropriate format for the target dataset.

        :param post_processed_sample: Processed training sample
        :type post_processed_sample: list[PostProcessedSample]
        :param sample_id: Original database sample identifier
        :type sample_id: int
        :param agent_id: Original database agent identifier
        :type agent_id: int
        :param sample_path: File path for the training sample
        :type sample_path: Path
        :param annotation_path: File path for the metadata/annotation
        :type annotation_path: Path

        :raises NotImplementedError: If not implemented in subclass
        :raises IOError: If file writing fails

        Example:
            .. code-block:: python

                def record_sample_and_annotation(self, sample, sample_id, agent_id, sample_path, ann_path):
                    # Create training format (e.g., Alpaca)
                    # Write sample JSON
                    # Write annotation JSON
        """
        pass

    def get_all_ids(self) -> list[int]:
        """
        Retrieve all sample IDs from the database.

        Queries the database to find all available samples for processing.
        This method is used to discover the complete set of data to process.

        :returns: List of sample IDs available in the database
        :rtype: list[int]

        :raises SQLAlchemyError: If database query fails

        Example:
            .. code-block:: python

                sample_ids = processor.get_all_ids()
                print(f"Found {len(sample_ids)} samples to process")
        """
        stmt = select(SamplesTable.id)
        result = self.session.execute(stmt).scalars().all()
        return result

    def process_all_samples_in_db(self):
        """
        Execute the complete dataset processing pipeline.

        This is the main entry point that orchestrates the entire dataset creation
        process. It handles directory preparation, sample discovery, data processing,
        and output generation with comprehensive logging and progress tracking.

        Processing steps:
            1. Prepare output directories with timestamp-based naming
            2. Discover all samples in the database
            3. Process each sample through the complete pipeline
            4. Save all processed data and metadata to files

        :raises DatabaseError: If sample discovery fails
        :raises IOError: If directory creation or file writing fails
        :raises ProcessingError: If data processing pipeline fails

        .. note::
            If the target directory already exists, a new timestamped directory
            will be created to avoid overwriting existing datasets.

        Example:
            .. code-block:: python

                processor = AlpacaDatasetV1(Path("./datasets"), None, session)
                processor.process_all_samples_in_db()
                # Creates: ./datasets_20240816_170429_dataset_version_1/

        Logging Output:
            - Directory preparation information
            - Sample count discovery
            - Processing progress with tqdm progress bar
            - Save operation status
        """

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
        """
        Process a single sample through the complete data pipeline.

        Executes the full processing workflow for one sample, including all
        agents within that sample. The method coordinates extraction, conversion,
        post-processing, and data splitting operations.

        :param sample_id: Database identifier for the sample to process
        :type sample_id: int

        :returns: Tuple containing agent information and processed sample data
        :rtype: tuple[AgentTable, PostProcessedSample]

        :raises AssertionError: If agent data structure is invalid
        :raises ProcessingError: If any pipeline step fails

        Processing workflow:
            1. Retrieve agent information for the sample
            2. Run all configured processors on the agent data
            3. Apply post-processing transformations
            4. Split data into training sequences

        .. note::
            Currently processes only the first agent in each sample. This may
            be extended in future versions to handle multiple agents per sample.

        Example:
            .. code-block:: python

                agent_info, processed_sample = processor.process_sample(sample_id=1)
                print(f"Processed agent {agent_info.agent_no}")
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
        """
        Prepare output directories for dataset files.

        Creates the necessary directory structure for saving processed samples
        and annotations. If the target directory already exists, creates a new
        timestamped directory to prevent data overwrites.

        :returns: Tuple of (samples_directory_path, annotations_directory_path)
        :rtype: tuple[Path, Path]

        :raises OSError: If directory creation fails
        :raises PermissionError: If insufficient permissions for directory creation

        Directory structure created:
            - `{data_dir}/samples/` - For training sample files
            - `{data_dir}/annotations/` - For metadata files

        Naming convention:
            - If directory exists: `{data_dir}_{timestamp}_dataset_version_{version}`
            - If new directory: `{data_dir}`

        Example:
            .. code-block:: python

                samples_path, annotations_path = processor._prepare_dir()
                # Returns: (Path("dataset/samples"), Path("dataset/annotations"))
        """
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
        """
        Retrieve agent information for a specific sample.

        Queries the database to get all agents associated with the given sample.
        This information is used to identify which agents to process and their
        characteristics.

        :param sample_id: Database identifier for the sample
        :type sample_id: int

        :returns: List of agent records wrapped in SQLAlchemy result tuples
        :rtype: list[tuple[AgentTable]]

        :raises SQLAlchemyError: If database query fails

        .. note::
            The returned agents are wrapped in tuples due to SQLAlchemy's
            fetchall() behavior. Extract agent objects using ``agent[0]``.

        Example:
            .. code-block:: python

                agents = processor._get_info_about_agents(sample_id=1)
                for agent_tuple in agents:
                    agent = agent_tuple[0]  # Extract AgentTable object
                    print(f"Agent {agent.agent_no}: {agent.role}")
        """
        stmt = select(AgentTable).where(AgentTable.sample_id == sample_id)
        return self.session.execute(stmt).fetchall()
