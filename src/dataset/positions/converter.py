from src.dataset.base.converters import BaseConverter
from src.dataset.base.models import (
    TimelineData,
    FetchedPositionsModel,
    FetchedPositionsT,
)
from src.database.db import Positions, PositionT
from src.dataset.base.tokens import TokensMapping


class PositionConverter(BaseConverter[Positions]):
    def convert(self, data_from_db: FetchedPositionsModel) -> TimelineData:
        ego_pos: FetchedPositionsT = data_from_db.ego_pos
        target_pos: FetchedPositionsT = data_from_db.target_pos

        processed_ego_positions = [self._convert_positions(pos) for pos in ego_pos]

        processed_target_positions = [
            self._convert_positions(pos) for pos in target_pos
        ]

        return processed_ego_positions + processed_target_positions

    def _convert_positions(self, row: tuple[Positions]) -> TimelineData:
        row = row[0]

        if row.type == PositionT.AGENT:
            pos_type = TokensMapping.EGO_POS_TOKEN.value
        elif row.type == PositionT.TARGET:
            pos_type = TokensMapping.TARGET_POS_TOKEN.value
        else:
            raise ValueError(f"Unknown position type: {row.type}. ")

        parts = [pos_type, str(row.pos_x), str(row.pos_y)]

        formatted = " ".join(parts)

        return TimelineData(
            timestamp=row.timestamp,
            formatted=formatted,
            type="position",
        )
