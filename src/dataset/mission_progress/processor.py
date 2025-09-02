from src.dataset.base.processors import BaseProcessor
from src.dataset.base.models import TimelineData
from src.dataset.mission_progress.converter import MissionProgressConverter
from src.dataset.mission_progress.extractor import MissionProgressExtractor


class DefaultMissionProgressProcessor(
    BaseProcessor[MissionProgressConverter, MissionProgressExtractor, TimelineData]
):
    def __init__(
        self, converter: MissionProgressConverter, extractor: MissionProgressExtractor
    ):

        if not isinstance(converter, MissionProgressConverter):
            raise ValueError(
                "Please, initialize the `DefaultMissionProgressProcessor` with `MissionProgressConverter`"
            )

        if not isinstance(extractor, MissionProgressExtractor):
            raise ValueError(
                "Please, initialize the `DefaultMissionProgressProcessor` with `MissionProgressExtractor`"
            )

        super().__init__(converter, extractor)

    def process(self, sample_id: int, agent_id: int) -> list[TimelineData]:
        return self.convert_all(sample_id=sample_id, agent_id=agent_id)
