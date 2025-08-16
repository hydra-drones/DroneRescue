import json
from pathlib import Path
from loguru import logger

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from src.database.db import (
    AgentTable,
    SamplesTable,
    Positions,
    Messages,
    Strategy,
    MissionProgress,
)
from src.database.db import PositionT, StrategyT
from src.data_processing.json_sample_model import (
    AgentPositionT,
    JSONSampleModel,
    TargetInFovT,
    LocalStrategyT,
    GlobalStrategyT,
    MissionProgressT,
    SentMessagesT,
)
from src.database.scripts import connect_to_db


class DataLoader:
    """Dataloader"""

    def __init__(self, session: Session):
        self.session = session

    def add_sample_to_db(self, json_path: Path):
        """Loads the data from JSON sample into Database

        Args:
            json_path (Path): JSON file with data from annotation App

        Returns:
            Bool: True if the sample was processed succesfully

        """
        try:
            sample_hash = SamplesTable.file_hash(json_path)
            existing_sample = (
                self.session.query(SamplesTable).filter_by(hash=sample_hash).first()
            )

            if existing_sample:
                logger.info(f"Sample {json_path} already exists in database")
                return False

            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)

            model = JSONSampleModel.model_validate(json_data)

            sample = SamplesTable(hash=sample_hash)
            self.session.add(sample)
            self.session.flush()

            self._load_agents(sample, model)

            self.session.commit()
            logger.info(f"Successfully loaded {json_path} with sample_id={sample.id}")
            return True

        except IntegrityError:
            self.session.rollback()
            logger.warning(f"Sample {json_path} already exists (integrity constraint)")
            return False
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to load {json_path}: {e}")
            raise

    def _load_agents(self, sample: SamplesTable, model: JSONSampleModel):
        """Load agents and their related data."""
        agent_map = {}  # Map from agent_no to Agent object

        for agent_no, agent_data in model.agents.items():
            agent = AgentTable(
                sample=sample,
                agent_no=int(agent_no),
                role=agent_data.role,
                mission=agent_data.mission,
            )
            self.session.add(agent)
            self.session.flush()
            agent_map[int(agent_no)] = agent

        for agent_no, agent_data in model.agents.items():

            agent = agent_map[int(agent_no)]

            self._load_agent_positions(sample.id, agent, agent_data.positions)
            self._load_target_positions(sample.id, agent, agent_data.target_in_fov)
            self._load_strategies(
                sample.id, agent, agent_data.global_strategy, StrategyT.GLOBAL
            )
            self._load_strategies(
                sample.id, agent, agent_data.local_strategy, StrategyT.LOCAL
            )
            self._load_mission_progress(sample.id, agent, agent_data.mission_progress)

            self._load_sent_messages(
                sample.id, agent, agent_data.sent_messages, agent_map
            )

    def _load_sent_messages(
        self,
        sample_id,
        agent,
        sent_messages: SentMessagesT,
        agent_map: dict[int, AgentTable],
    ):
        for timestamp, sent_message in sent_messages.items():
            for message in sent_message:
                msg = Messages(
                    sample_id=sample_id,
                    timestamp=int(timestamp),
                    sender=agent,
                    receiver=agent_map[message.receiver_id],
                    message=message.message,
                    type=message.type,
                )

                self.session.add(msg)

    def _load_agent_positions(
        self,
        sample_id: int,
        agent_db: AgentTable,
        positions: AgentPositionT,
    ):
        """Load position data."""
        for timestamp, position in positions.items():
            pos = Positions(
                sample_id=sample_id,
                agent=agent_db,
                timestamp=int(timestamp),
                pos_x=position.pos_x,
                pos_y=position.pos_y,
                type=position.type,
            )
            self.session.add(pos)

    def _load_target_positions(
        self,
        sample_id: int,
        agent_db: AgentTable,
        targets_in_fov: TargetInFovT,
    ):
        """Load target positions in field of view."""
        for timestamp, targets in targets_in_fov.items():
            for target_pos in targets:
                pos = Positions(
                    sample_id=sample_id,
                    agent=agent_db,
                    timestamp=int(timestamp),
                    pos_x=target_pos.pos_x,
                    pos_y=target_pos.pos_y,
                    type=PositionT.TARGET,
                )
                self.session.add(pos)

    def _load_strategies(
        self,
        sampled_id: int,
        agent_db: AgentTable,
        strategies: GlobalStrategyT | LocalStrategyT,
        strategy_type: StrategyT,
    ):
        """Load strategy data."""
        for timestamp, strategy_text in strategies.items():
            strategy = Strategy(
                sample_id=sampled_id,
                agent=agent_db,
                timestamp=int(timestamp),
                strategy=strategy_text,
                type=strategy_type,
            )
            self.session.add(strategy)

    def _load_mission_progress(
        self, sample_id: int, agent_db: AgentTable, progress_data: MissionProgressT
    ):
        """Load mission progress data."""
        for timestamp, progress_text in progress_data.items():
            progress = MissionProgress(
                sample_id=sample_id,
                agent=agent_db,
                timestamp=int(timestamp),
                progress=progress_text,
            )
            self.session.add(progress)


def load_dataset_to_db(
    dataset_dir: Path = Path("datasamples/"), db_path: Path = Path(".database/data.db")
):
    """Load all JSON files from dataset into database."""

    if not dataset_dir.exists():
        logger.error(f"Dataset directory {dataset_dir} does not exist.")
        return

    session = connect_to_db(db_path)

    try:
        loader = DataLoader(session)

        for sample in dataset_dir.iterdir():
            if not (sample.suffix == ".json"):
                continue

            logger.info(f"Processing {sample}...")
            loader.add_sample_to_db(sample)

    finally:
        session.close()


if __name__ == "__main__":
    load_dataset_to_db()
