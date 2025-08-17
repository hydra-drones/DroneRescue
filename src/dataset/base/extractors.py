from abc import ABC, abstractmethod
from typing import Generic
from sqlalchemy.orm import Session
from src.dataset.base.models import D


class BaseExtractor(ABC, Generic[D]):
    """
    Abstract base class for extracting data from database sources.

    This class provides a generic interface for fetching data from various database
    tables and preparing it for conversion into structured timeline data. Each
    extractor is responsible for executing the appropriate database queries and
    returning the results in a standardized format.

    :param D: Type parameter representing the database model type (e.g., Messages, Positions)
    :type D: TypeVar

    :param session: SQLAlchemy database session for executing queries
    :type session: Session

    .. note::
        All concrete extractor implementations must inherit from this class and implement
        the :meth:`fetch_data` method with appropriate query logic.

    Example:
        .. code-block:: python

            class MessageExtractor(BaseExtractor[Messages]):
                def fetch_data(self, sample_id, agent_id):
                    # Implementation specific to Messages table
                    return fetched_data

    .. seealso::
        :class:`~src.dataset.message.extractor.MessageExtractor`
            Concrete implementation for extracting message data
    """

    def __init__(self, session: Session):
        """
        Initialize the extractor with a database session.

        :param session: SQLAlchemy session for database operations
        :type session: Session

        Example:
            .. code-block:: python

                from sqlalchemy.orm import Session
                extractor = MessageExtractor(session=db_session)
        """
        super().__init__()
        self.session = session

    @abstractmethod
    def fetch_data(self, **kwargs) -> list[D] | tuple[list[D], list[D]]:
        """
        Fetch data from the database based on provided parameters.

        This method should implement the specific query logic for extracting
        data from the database. The exact parameters and return format depend
        on the implementing subclass and the type of data being extracted.

        :param kwargs: Variable keyword arguments specific to the data type being extracted
        :type kwargs: dict

        :returns: Extracted data in one of two formats:
                 - A list of database records of type D
                 - A tuple containing two lists of database records (for related data)
        :rtype: list[D] | tuple[list[D], list[D]]

        :raises NotImplementedError: If the method is not implemented in subclass
        :raises ValueError: If the provided parameters are invalid
        :raises SQLAlchemyError: If database query fails

        .. note::
            The specific parameters required depend on the implementing subclass.
            Common parameters include sample_id, agent_id, timestamp ranges, etc.

        Example:
            .. code-block:: python

                extractor = MessageExtractor(session=db_session)
                data = extractor.fetch_data(sample_id=1, agent_id=5)
        """
        pass
