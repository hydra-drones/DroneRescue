from src.dataset.base.converters import BaseConverter
from src.dataset.base.models import (
    TimelineData,
    FetchedMisionProgressModel,
    FetchedMissionProgressT,
)
from src.database.db import MissionProgress
from src.dataset.base.tokens import TokensMapping


class MissionProgressConverter(BaseConverter[MissionProgress]):
    def convert(self, data_from_db: FetchedMisionProgressModel) -> TimelineData:
        mission_progresses: FetchedMissionProgressT = data_from_db.mission_progress

        processed_mission_progresses = [
            self._convert_mission_progress(m_pgr) for m_pgr in mission_progresses
        ]

        return processed_mission_progresses

    def _convert_mission_progress(self, row: tuple[MissionProgress]) -> TimelineData:
        row: MissionProgress = row[0]

        parts = [
            TokensMapping.MISSION_PROGRESS_TOKEN.value,
            str(row.progress),
        ]

        formatted = " ".join(parts)

        return TimelineData(
            timestamp=row.timestamp,
            formatted=formatted,
            type="mission_progress",
        )
