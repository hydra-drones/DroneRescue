from src.dataset.base.converters import BaseConverter
from src.dataset.base.models import (
    TimelineData,
    FetchedStrategyModel,
    FetchedStrategyT,
)
from src.database.db import Strategy, StrategyT
from src.dataset.base.tokens import TokensMapping


class StrategyConverter(BaseConverter[Strategy]):
    def convert(self, data_from_db: FetchedStrategyModel) -> TimelineData:
        local_strategies: FetchedStrategyT = data_from_db.local_strategy
        global_strategies: FetchedStrategyT = data_from_db.global_strategy

        processed_local_strategies = [
            self._convert_strategy(strategy) for strategy in local_strategies
        ]

        processed_global_strategies = [
            self._convert_strategy(strategy) for strategy in global_strategies
        ]

        return processed_local_strategies + processed_global_strategies

    def _convert_strategy(self, row: tuple[Strategy]) -> TimelineData:
        row: Strategy = row[0]

        if row.type == StrategyT.LOCAL:
            strategy_type_token = TokensMapping.LOCAL_STRATEGY_TOKEN.value
        elif row.type == StrategyT.GLOBAL:
            strategy_type_token = TokensMapping.GLOBAL_STRATEGY_TOKEN.value
        else:
            raise ValueError(f"Unknown strategy type: {row.type}. ")

        parts = [strategy_type_token, row.strategy]

        formatted = " ".join(parts)

        return TimelineData(
            timestamp=row.timestamp,
            formatted=formatted,
            type="strategy",
        )
