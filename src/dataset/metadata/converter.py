from src.dataset.base.converters import BaseConverter
from src.dataset.base.models import (
    TimelineData,
    FetchedMetadataModel,
    FetchedMetadataT,
)
from src.database.db import AgentTable, AgentRoles
from src.dataset.base.tokens import TokensMapping


class MetadataConverter(BaseConverter[AgentTable]):

    HARDCODED_TIMESTAMP: int = 0

    def convert(self, data_from_db: FetchedMetadataModel) -> TimelineData:
        metadatas: FetchedMetadataT = data_from_db.metadata

        processed_metadata = [
            self._convert_metadata(metadata) for metadata in metadatas
        ]

        return processed_metadata

    def _convert_metadata(self, row: tuple[AgentTable]) -> TimelineData:
        row: AgentTable = row[0]

        if row.role.value == AgentRoles.SCOUT.value:
            agent_role = TokensMapping.SCOUT_TOKEN.value
        elif row.role.value == AgentRoles.RESCUER.value:
            agent_role = TokensMapping.RESCUER_TOKEN.value
        elif row.role.value == AgentRoles.COMMANDER.value:
            agent_role = TokensMapping.COMMANDER_TOKEN.value
        else:
            raise ValueError(f"Unknown agent role: {row.role.value}.")

        parts = [
            TokensMapping.START_METADATA.value,
            TokensMapping.AGENT_NUMBER.value,
            TokensMapping.AGENT_TYPE.value,
            agent_role,
            TokensMapping.AGENT_NUMBER.value,
            str(row.agent_no),
            TokensMapping.MAIN_MISSION_TOKEN.value,
            row.mission,
            TokensMapping.END_METADATA.value,
        ]

        formatted = " ".join(parts)

        return TimelineData(
            timestamp=self.HARDCODED_TIMESTAMP,
            formatted=formatted,
            type="metadata",
        )
