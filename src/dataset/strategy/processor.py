from src.dataset.base.processors import BaseProcessor
from src.dataset.base.models import TimelineData
from src.dataset.strategy.converter import StrategyConverter
from src.dataset.strategy.extractor import StrategyExtractor


class DefaultStrategyProcessor(
    BaseProcessor[StrategyConverter, StrategyExtractor, TimelineData]
):
    def __init__(self, converter: StrategyConverter, extractor: StrategyExtractor):

        if not isinstance(converter, StrategyConverter):
            raise ValueError(
                "Please, initialize the `DefaultStrategyProcessor` with `StrategyConverter`"
            )

        if not isinstance(extractor, StrategyExtractor):
            raise ValueError(
                "Please, initialize the `DefaultStrategyProcessor` with `StrategyExtractor`"
            )

        super().__init__(converter, extractor)

    def process(self, sample_id: int, agent_id: int) -> list[TimelineData]:
        return self.convert_all(sample_id=sample_id, agent_id=agent_id)
