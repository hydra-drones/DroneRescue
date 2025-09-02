from sqlalchemy import select

from src.dataset.base.extractors import BaseExtractor
from src.database.db import StrategyT, Strategy
from src.dataset.base.models import FetchedStrategyModel


class StrategyExtractor(BaseExtractor[Strategy]):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fetch_data(self, sample_id: int, agent_id: int) -> FetchedStrategyModel:

        local_strategy = self.session.execute(
            select(Strategy).where(
                (Strategy.sample_id == sample_id)
                & (Strategy.agent_id == agent_id)
                & (Strategy.type == StrategyT.LOCAL)
            )
        ).fetchall()

        global_strategy = self.session.execute(
            select(Strategy).where(
                (Strategy.sample_id == sample_id)
                & (Strategy.agent_id == agent_id)
                & (Strategy.type == StrategyT.GLOBAL)
            )
        ).fetchall()

        return FetchedStrategyModel(
            local_strategy=local_strategy, global_strategy=global_strategy
        )
