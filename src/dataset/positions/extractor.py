from sqlalchemy import select
from sqlalchemy.orm import selectinload

from src.dataset.base.extractors import BaseExtractor
from src.dataset.base.models import FetchedPositionsModel
from src.database.db import PositionT, Positions


class PositionExtractor(BaseExtractor[Positions]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_data(self, sample_id: int, agent_id: int) -> FetchedPositionsModel:

        ego_position = self.session.execute(
            select(Positions).where(
                (Positions.sample_id == sample_id)
                & (Positions.agent_id == agent_id)
                & (Positions.type == PositionT.AGENT)
            )
        ).fetchall()

        target_position = self.session.execute(
            select(Positions).where(
                (Positions.sample_id == sample_id)
                & (Positions.agent_id == agent_id)
                & (Positions.type == PositionT.TARGET)
            )
        ).fetchall()

        return FetchedPositionsModel(ego_pos=ego_position, target_pos=target_position)
