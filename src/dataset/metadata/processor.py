from src.dataset.base.processors import BaseProcessor
from src.dataset.base.models import TimelineData
from src.dataset.metadata.converter import MetadataConverter
from src.dataset.metadata.extractor import MetadataExtractor


class DefaultMetadataProcessor(
    BaseProcessor[MetadataConverter, MetadataExtractor, TimelineData]
):
    def __init__(self, converter: MetadataConverter, extractor: MetadataExtractor):

        if not isinstance(converter, MetadataConverter):
            raise ValueError(
                "Please, initialize the `DefaultMetadataProcessor` with `MetadataConverter`"
            )

        if not isinstance(extractor, MetadataExtractor):
            raise ValueError(
                "Please, initialize the `DefaultMetadataProcessor` with `PositionExtractor`"
            )

        super().__init__(converter, extractor)

    def process(self, sample_id: int, agent_id: int) -> list[TimelineData]:
        return self.convert_all(sample_id=sample_id, agent_id=agent_id)
