from src.dataset.base.processors import BaseProcessor
from src.dataset.base.models import TimelineData
from src.dataset.message.converter import MessageConverter
from src.dataset.message.extractor import MessageExtractor


class DefaultMessageProcessor(
    BaseProcessor[MessageConverter, MessageExtractor, TimelineData]
):
    """
    Default implementation for processing message data.

    This processor combines message extraction and conversion to transform
    raw message data from the database into structured timeline format suitable
    for machine learning training. It handles both sent and received messages
    for a given agent within a specific sample.

    The processor uses:
        - :class:`~src.dataset.message.extractor.MessageExtractor` for data extraction
        - :class:`~src.dataset.message.converter.MessageConverter` for data transformation

    Processing workflow:
        1. Extract sent and received messages from database
        2. Convert messages to standardized timeline format
        3. Apply token processing (agent tokens, position tokens)
        4. Return formatted timeline data

    :inherits: :class:`~src.dataset.base.processors.BaseProcessor`

    Example:
        .. code-block:: python

            processor = DefaultMessageProcessor(
                converter=MessageConverter(),
                extractor=MessageExtractor(session=db_session)
            )

            timeline_data = processor.process(sample_id=1, agent_id=5)
            print(f"Processed {len(timeline_data)} messages")

    .. seealso::
        :class:`~src.dataset.message.converter.MessageConverter`
            Message data conversion component
        :class:`~src.dataset.message.extractor.MessageExtractor`
            Message data extraction component
    """

    def __init__(self, converter: MessageConverter, extractor: MessageExtractor):
        """
        Initialize the message processor with converter and extractor.

        :param converter: Message data converter instance
        :type converter: MessageConverter
        :param extractor: Message data extractor instance
        :type extractor: MessageExtractor

        Example:
            .. code-block:: python

                processor = DefaultMessageProcessor(
                    converter=MessageConverter(),
                    extractor=MessageExtractor(session=db_session)
                )
        """
        super().__init__(converter, extractor)

    def process(self, sample_id: int, agent_id: int) -> list[TimelineData]:
        """
        Process message data for a specific agent and sample.

        Executes the complete message processing pipeline by extracting message
        data from the database and converting it to timeline format. This method
        coordinates the extractor and converter components to produce training-ready data.

        :param sample_id: Database identifier for the sample/simulation run
        :type sample_id: int
        :param agent_id: Database identifier for the agent
        :type agent_id: int

        :returns: List of processed timeline data entries for all messages
        :rtype: list[TimelineData]

        :raises DatabaseError: If message extraction fails
        :raises ConversionError: If message conversion fails

        Output format:
            Each returned :class:`~src.dataset.base.models.TimelineData` contains:
                - **timestamp**: Message timestamp for chronological ordering
                - **formatted**: Tokenized message content ready for ML training
                - **type**: Message direction ("sent_message" or "recieved_message")

        Example:
            .. code-block:: python

                timeline_data = processor.process(sample_id=1, agent_id=5)

                for entry in timeline_data:
                    print(f"{entry.timestamp}: {entry.type}")
                    print(f"Content: {entry.formatted}")

                # Example output:
                # 1001: sent_message
                # Content: <SND> <TO> <AGENT#3> <ORDER> <MESSAGE> Deploy to <POS> 10 20
                # 1005: recieved_message
                # Content: <RCV> <AGENT#2> <TOME> <INFO> <MESSAGE> Roger, moving to position

        .. note::
            This method delegates to :meth:`convert_all` which handles the
            extraction and conversion pipeline automatically.
        """
        return self.convert_all(sample_id=sample_id, agent_id=agent_id)
