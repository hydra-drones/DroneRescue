from sqlalchemy import select

from src.dataset.base.extractors import BaseExtractor
from src.dataset.base.models import FetchedMetadataModel
from src.database.db import AgentTable


class MetadataExtractor(BaseExtractor[AgentTable]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_data(self, sample_id: int, agent_id: int) -> FetchedMetadataModel:

        mission_progress = self.session.execute(
            select(AgentTable).where((AgentTable.sample_id == sample_id))
        ).fetchall()

        return FetchedMetadataModel(metadata=mission_progress)
