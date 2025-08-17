from src.dataset.base.converters import BaseConverter
from src.dataset.base.models import TimelineData, FetchedMessagesModel, FetchedMessagesT
from src.database.db import Messages, MessageT
from src.dataset.base.tokens import TokensMapping


class MessageConverter(BaseConverter[Messages]):
    """
    Converter for transforming message data into structured timeline format.

    This class handles the conversion of raw message data (both sent and received)
    from the database into formatted :class:`~src.dataset.base.models.TimelineData`
    objects suitable for language model training. It applies tokenization, formatting,
    and agent token processing to create consistent training data.

    The converter processes two types of messages:
        - **Sent messages**: Messages sent by the current agent to other agents
        - **Received messages**: Messages received by the current agent from other agents

    Each message is converted to include:
        - Appropriate directional tokens (SENT/RECEIVED)
        - Agent identification tokens
        - Message type tokens (INFO/ORDER)
        - Processed message content with agent and position tokens

    :inherits: :class:`~src.dataset.base.converters.BaseConverter`

    Example:
        .. code-block:: python

            converter = MessageConverter()
            fetched_data = FetchedMessagesModel(sent_msgs, received_msgs)
            timeline_data = converter.convert(fetched_data)

    .. seealso::
        :class:`~src.dataset.base.models.FetchedMessagesModel`
            Input data structure for message conversion
        :class:`~src.dataset.base.models.TimelineData`
            Output data structure after conversion
    """

    def convert(self, data_from_db: FetchedMessagesModel) -> TimelineData:
        """
        Convert fetched message data into formatted timeline data.

        Processes both sent and received messages from the database, applying
        appropriate tokenization and formatting to create training-ready data
        for language models.

        :param data_from_db: Container with sent and received message data
        :type data_from_db: FetchedMessagesModel

        :returns: List of processed timeline data entries, one for each message
        :rtype: list[TimelineData]

        :raises ValueError: If message type is not recognized (not INFO or ORDER)

        .. note::
            The returned list contains both sent and received messages in the order
            they were processed, not necessarily chronological order.

        Example:
            .. code-block:: python

                fetched_msgs = FetchedMessagesModel(sent_msgs, received_msgs)
                timeline_data = converter.convert(fetched_msgs)
                # Returns: [TimelineData(...), TimelineData(...), ...]
        """
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
        """
        Convert a single sent message into formatted timeline data.

        Processes a sent message by formatting it with appropriate tokens including
        sender identification, receiver agent tokens, message type, and processed
        message content.

        :param row: Database row containing the message data (wrapped in tuple)
        :type row: tuple[Messages]

        :returns: Formatted timeline data for the sent message
        :rtype: TimelineData

        :raises ValueError: If the message type is not INFO or ORDER
        :raises AttributeError: If receiver agent information is not available

        .. note::
            The formatted output follows the pattern:
            ``<SND> <TO> <AGENT#X> <INFO|ORDER> <MESSAGE> processed_content``

        Example:
            Input message: "Deploy to position (10, 20)"
            Output format: "<SND> <TO> <AGENT#5> <ORDER> <MESSAGE> Deploy to <POS> 10 20"
        """
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
            TokensMapping.get_agent_token(receiver_agent_no),
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
        """
        Convert a single received message into formatted timeline data.

        Processes a received message by formatting it with appropriate tokens including
        sender agent identification, "to me" indicator, message type, and processed
        message content.

        :param row: Database row containing the message data (wrapped in tuple)
        :type row: tuple[Messages]

        :returns: Formatted timeline data for the received message
        :rtype: TimelineData

        :raises ValueError: If the message type is not INFO or ORDER
        :raises AttributeError: If sender agent information is not available

        .. note::
            The formatted output follows the pattern:
            ``<RCV> AGENT#X <TOME> <INFO|ORDER> <MESSAGE> processed_content``

        Example:
            Input message: "Move AGENT#3 to sector B"
            Output format: "<RCV> AGENT#2 <TOME> <ORDER> <MESSAGE> Move <AGENT#3> to sector B"
        """
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
            TokensMapping.get_agent_token(sender_agent_no),
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
        """
        Apply token processing to raw message content.

        Transforms natural language message content by replacing specific patterns
        with standardized tokens. This includes converting agent references and
        position coordinates to their tokenized equivalents.

        :param message: Raw message content from the database
        :type message: str

        :returns: Processed message with tokens applied
        :rtype: str

        Processing steps:
            1. Convert agent references (e.g., "Scout 3" → "AGENT#3")
            2. Convert position coordinates (e.g., "(10, 20)" → "<POS> 10 20")

        Example:
            .. code-block:: python

                raw_msg = "Scout 10 move to position (8, 90)"
                processed = self._process_message_with_tokens(raw_msg)
                # Result: "<AGENT#10> move to position <POS> 8 90"

        .. seealso::
            :meth:`~src.dataset.base.tokens.TokensMapping.process_message_with_agent_tokens`
                Agent token processing
            :meth:`~src.dataset.base.tokens.TokensMapping.process_message_with_position_tokens`
                Position token processing
        """
        processed_message = TokensMapping.process_message_with_agent_tokens(message)
        processed_message = TokensMapping.process_message_with_position_tokens(
            processed_message
        )
        return processed_message
