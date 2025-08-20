from sqlalchemy import select

from src.dataset.base.extractors import BaseExtractor
from src.dataset.base.models import FetchedMisionProgressModel
from src.database.db import MissionProgress


class MissionProgressExtractor(BaseExtractor[MissionProgress]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_data(self, sample_id: int, agent_id: int) -> FetchedMisionProgressModel:

        mission_progress = self.session.execute(
            select(MissionProgress).where(
                (MissionProgress.sample_id == sample_id)
                & (MissionProgress.agent_id == agent_id)
            )
        ).fetchall()

        return FetchedMisionProgressModel(mission_progress=mission_progress)
