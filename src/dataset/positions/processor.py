from src.dataset.base.processors import BaseProcessor
from src.dataset.base.models import TimelineData
from src.dataset.positions.converter import PositionConverter
from src.dataset.positions.extractor import PositionExtractor


class DefaultPositionProcessor(
    BaseProcessor[PositionConverter, PositionExtractor, TimelineData]
):
    def __init__(self, converter: PositionConverter, extractor: PositionExtractor):

        if not isinstance(converter, PositionConverter):
            raise ValueError(
                "Please, initialize the `DefaultPositionProcessor` with `PositionConverter`"
            )

        if not isinstance(extractor, PositionExtractor):
            raise ValueError(
                "Please, initialize the `DefaultPositionProcessor` with `PositionExtractor`"
            )

        super().__init__(converter, extractor)

    def process(self, sample_id: int, agent_id: int) -> list[TimelineData]:
        return self.convert_all(sample_id=sample_id, agent_id=agent_id)
