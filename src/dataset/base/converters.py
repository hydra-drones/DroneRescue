from abc import abstractmethod, ABC
from typing import Generic
from src.dataset.base.models import D, TimelineData


class BaseConverter(ABC, Generic[D]):
    """
    Abstract base class for converting raw database data into structured timeline data.

    This class provides a generic interface for transforming extracted database records
    into standardized :class:`~src.dataset.base.models.TimelineData` objects that can
    be used for machine learning training and analysis.

    :param D: Type parameter representing the database model type (e.g., Messages, Positions)
    :type D: TypeVar

    .. note::
        All concrete converter implementations must inherit from this class and implement
        the :meth:`convert` method.

    Example:
        .. code-block:: python

            class MessageConverter(BaseConverter[Messages]):
                def convert(self, data_from_db):
                    # Implementation specific to Messages
                    return timeline_data_list

    .. seealso::
        :class:`~src.dataset.message.converter.MessageConverter`
            Concrete implementation for converting message data
    """

    @abstractmethod
    def convert(self, data_from_db: list[D] | tuple[list[D], list[D]]) -> TimelineData:
        """
        Convert raw database data into structured timeline data.

        This method transforms extracted database records into a standardized format
        that can be used for training machine learning models or further processing.

        :param data_from_db: Raw data extracted from database. Can be either:
                           - A list of database records of type D
                           - A tuple containing two lists of database records
        :type data_from_db: list[D] | tuple[list[D], list[D]]

        :returns: Processed timeline data ready for ML training or analysis
        :rtype: TimelineData

        :raises NotImplementedError: If the method is not implemented in subclass
        :raises ValueError: If the input data format is invalid or incompatible

        .. note::
            The exact transformation logic depends on the specific data type and
            requirements of the implementing subclass.

        Example:
            .. code-block:: python

                converter = MessageConverter()
                timeline_data = converter.convert(fetched_messages)
        """
        pass
