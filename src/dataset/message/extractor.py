from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.dataset.base.extractors import BaseExtractor
from src.dataset.base.models import FetchedMessagesModel
from src.database.db import Messages


class MessageExtractor(BaseExtractor[Messages]):
    """
    Extractor for fetching message data from the database.

    This class handles the extraction of both sent and received messages for a specific
    agent within a given sample. It performs optimized database queries using SQLAlchemy
    with eager loading of related agent data to minimize database round trips.

    The extractor retrieves two categories of messages:
        - **Sent messages**: Messages where the specified agent is the sender
        - **Received messages**: Messages where the specified agent is the receiver

    Related agent data (sender/receiver information) is eagerly loaded to support
    subsequent processing steps that require agent details.

    :inherits: :class:`~src.dataset.base.extractors.BaseExtractor`

    Example:
        .. code-block:: python

            extractor = MessageExtractor(session=db_session)
            messages = extractor.fetch_data(sample_id=1, agent_id=5)
            print(f"Sent: {len(messages.sent_messages)}")
            print(f"Received: {len(messages.recieved_messages)}")

    .. seealso::
        :class:`~src.dataset.base.models.FetchedMessagesModel`
            Container for the extracted message data
        :class:`~src.database.db.Messages`
            Database model for message records
    """

    def __init__(self, **kwargs):
        """
        Initialize the message extractor.

        :param kwargs: Keyword arguments passed to the parent BaseExtractor,
                      typically including the database session
        :type kwargs: dict

        Example:
            .. code-block:: python

                extractor = MessageExtractor(session=db_session)
        """
        super().__init__(**kwargs)

    def fetch_data(self, sample_id: int, agent_id: int) -> FetchedMessagesModel:
        """
        Extract all message data for a specific agent within a sample.

        Performs optimized database queries to retrieve both sent and received messages
        for the specified agent. Uses eager loading to include related agent information
        (sender/receiver details) to support downstream processing without additional
        database queries.

        :param sample_id: Unique identifier of the sample/simulation run
        :type sample_id: int
        :param agent_id: Database ID of the agent whose messages to extract
        :type agent_id: int

        :returns: Container with separated sent and received message collections
        :rtype: FetchedMessagesModel

        :raises SQLAlchemyError: If database queries fail
        :raises ValueError: If sample_id or agent_id are invalid

        Query Details:
            - **Sent messages**: ``Messages.sender_id == agent_id``
            - **Received messages**: ``Messages.receiver_id == agent_id``
            - **Eager loading**: Related agent data for efficient access

        Database Relationships Loaded:
            - For sent messages: receiver agent information
            - For received messages: sender agent information

        Example:
            .. code-block:: python

                extractor = MessageExtractor(session=db_session)
                messages = extractor.fetch_data(sample_id=1, agent_id=5)

                # Access sent messages
                for row in messages.sent_messages:
                    msg = row[0]  # Extract Messages object
                    receiver = msg.receiver.agent_no  # Eagerly loaded

                # Access received messages
                for row in messages.recieved_messages:
                    msg = row[0]  # Extract Messages object
                    sender = msg.sender.agent_no  # Eagerly loaded

        .. note::
            The returned message rows are wrapped in tuples due to SQLAlchemy's
            fetchall() behavior. Extract the Messages object using ``row[0]``.

        .. seealso::
            :class:`~src.dataset.base.models.FetchedMessagesModel`
                Return type containing sent and received messages
            :meth:`~sqlalchemy.orm.selectinload`
                SQLAlchemy eager loading technique used
        """

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
