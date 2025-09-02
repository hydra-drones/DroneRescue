from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import DateTime
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from hashlib import sha256
from pathlib import Path
from sqlalchemy.sql import functions
from loguru import logger
from sqlalchemy import event
from enum import Enum


class MessageT(Enum):
    ORDER = "order"
    INFO = "info"


class PositionT(Enum):
    AGENT = "agent"
    TARGET = "target"
    BASE = "base"


class StrategyT(Enum):
    LOCAL = "local"
    GLOBAL = "global"


class TableNames(str, Enum):
    SAMPLES = "samples"
    AGENT = "agent"
    MESSAGES = "messages"
    STRATEGY = "strategy"
    POSITIONS = "positions"
    MISSION_PROGRESS = "mission_progress"


class AgentRoles(Enum):
    COMMANDER = "scout_commander"
    SCOUT = "scout"
    RESCUER = "rescuer"


class Base(DeclarativeBase):
    pass


class AgentTable(Base):
    __tablename__ = TableNames.AGENT.value

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.SAMPLES.value}.id"))
    agent_no: Mapped[int] = mapped_column(Integer)
    role: Mapped[AgentRoles] = mapped_column(SQLEnum(AgentRoles), nullable=False)
    mission: Mapped[str] = mapped_column(String)

    sample: Mapped["SamplesTable"] = relationship(
        "SamplesTable", back_populates="agents"
    )
    messages_sent: Mapped[List["Messages"]] = relationship(
        "Messages",
        back_populates="sender",
        cascade="all, delete-orphan",
        foreign_keys="Messages.sender_id",
    )

    messages_received: Mapped[List["Messages"]] = relationship(
        "Messages",
        back_populates="receiver",
        cascade="all, delete-orphan",
        foreign_keys="Messages.receiver_id",
    )

    strategies: Mapped[List["Strategy"]] = relationship(
        "Strategy", back_populates="agent", cascade="all, delete-orphan"
    )

    positions: Mapped[List["Positions"]] = relationship(
        "Positions", back_populates="agent", cascade="all, delete-orphan"
    )

    mission_progresses: Mapped[List["MissionProgress"]] = relationship(
        "MissionProgress", back_populates="agent", cascade="all, delete-orphan"
    )


class SamplesTable(Base):
    """Contains meta information for samples from annotation app

    Columns:
        - id: primary column ; internal ID of sample, created automatically.
        - hash: hash is calculated for each individual JSON file with date.
        - date: date when the JSON has been added.
    """

    __tablename__ = TableNames.SAMPLES.value

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[str] = mapped_column(String, unique=True)
    created_at: Mapped[DateTime] = mapped_column(
        DateTime, server_default=functions.now()
    )

    agents = relationship(
        "AgentTable", back_populates="sample", cascade="all, delete-orphan"
    )

    @staticmethod
    def file_hash(path: Path) -> str:
        """Calculates SHA256 hash for the given JOSN file."""
        return sha256(path.read_bytes()).hexdigest()

    def __repr__(self) -> str:
        return (
            f"<{TableNames.SAMPLES.value} ("
            f"id={self.id}, "
            f"hash='{self.hash}', "
            f"created_at={self.created_at}"
            f")>"
        )


class Messages(Base):
    """Sent Messages from agent to another agent or agents

    Columns:
        - sample_id: id of the sample where the data comes from.
        - timestamp: internal timestamp when the message was sent.
        - sender_id: agent's id (ego).
        - receiver_id: the receiver of the message.
        - message: sent message.
        - type: type of the message ; can be "order" or "info".
    """

    __tablename__ = TableNames.MESSAGES.value

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.SAMPLES.value}.id"))
    timestamp: Mapped[int] = mapped_column(Integer)
    sender_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.AGENT.value}.id"))
    receiver_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.AGENT.value}.id"))
    message: Mapped[str] = mapped_column(String)
    type: Mapped[MessageT] = mapped_column(SQLEnum(MessageT), nullable=False)

    sender = relationship(
        "AgentTable", foreign_keys=[sender_id], back_populates="messages_sent"
    )
    receiver = relationship(
        "AgentTable", foreign_keys=[receiver_id], back_populates="messages_received"
    )

    def __repr__(self) -> str:
        preview = self.message
        if len(preview) > 30:
            preview = preview[:27] + "..."

        return (
            f"<{TableNames.MESSAGES.value} ("
            f"sample_id={self.sample_id}, "
            f"timestamp={self.timestamp}, "
            f"sender_id={self.sender_id}, "
            f"receiver_id={self.receiver_id}, "
            f"type='{self.type}', "
            f"message='{preview}'"
            f")>"
        )


class Strategy(Base):
    """Table for Local Strategy

    Columns:
        - id. Unique ID.
        - sample_id: ID of the sample.
        - agent_id: Agent's ID the changing of strategy belongs to.
        - timestamp: timestamp of the changing the strategy.
        - trategy: text of the new local strategy.

    """

    __tablename__ = TableNames.STRATEGY.value

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.SAMPLES.value}.id"))
    agent_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.AGENT.value}.id"))
    timestamp: Mapped[int] = mapped_column(Integer)
    strategy: Mapped[str] = mapped_column(String)
    type: Mapped[StrategyT] = mapped_column(SQLEnum(StrategyT), nullable=False)

    agent = relationship(
        "AgentTable", foreign_keys=[agent_id], back_populates="strategies"
    )

    def __repr__(self) -> str:
        preview = self.strategy
        if len(preview) > 30:
            preview = preview[:27] + "..."

        return (
            f"<{TableNames.STRATEGY.value} ("
            f"id={self.id}, "
            f"sample_id={self.sample_id}, "
            f"timestamp={self.timestamp}, "
            f"agent_id={self.agent_id}, "
            f"strategy='{preview}'"
            f")>"
        )


class Positions(Base):
    """Positions Table

    Contains positions (x, y) of the object.
    If :code:`type="agent"` it means the position of the agent.
    Otherwise, :code:`type="target"`, it means the position of the
    target in the field of view for the given agent.

    Columns:
    - id. Unique ID.
    - sample_id: ID of the sample.
    - agent_id: Agent's ID the changing of local strategy belongs to.
    - timestamp: timestamp of the changing the local strategy.
    - pos_x: X coordinate.
    - pos_y: Y coordinate.
    """

    __tablename__ = TableNames.POSITIONS.value

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.SAMPLES.value}.id"))
    timestamp: Mapped[int] = mapped_column(Integer)
    agent_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.AGENT.value}.id"))
    pos_x: Mapped[int] = mapped_column(Integer)
    pos_y: Mapped[int] = mapped_column(Integer)
    type: Mapped[PositionT] = mapped_column(SQLEnum(PositionT), nullable=False)

    agent = relationship(
        "AgentTable", foreign_keys=[agent_id], back_populates="positions"
    )

    def __repr__(self) -> str:
        return (
            f"<{TableNames.POSITIONS.value} ("
            f"id={self.id}, "
            f"sample_id={self.sample_id}, "
            f"timestamp={self.timestamp}, "
            f"agent_id={self.agent_id}, "
            f"pos_x={self.pos_x}, "
            f"pos_y={self.pos_y}, "
            f"type={self.type}, "
            f")>"
        )


class MissionProgress(Base):
    """Table for Local Strategy

    Columns:
        - id. Unique ID.
        - sample_id: ID of the sample.
        - agent_id: Agent's ID the changing of local strategy belongs to.
        - timestamp: timestamp of the changing the local strategy.
        - local_strategy: text of the new local strategy.

    """

    __tablename__ = TableNames.MISSION_PROGRESS.value

    id: Mapped[int] = mapped_column(primary_key=True)
    sample_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.SAMPLES.value}.id"))
    agent_id: Mapped[int] = mapped_column(ForeignKey(f"{TableNames.AGENT.value}.id"))
    timestamp: Mapped[int] = mapped_column(Integer)
    progress: Mapped[str] = mapped_column(String)

    agent = relationship(
        "AgentTable", foreign_keys=[agent_id], back_populates="mission_progresses"
    )

    def __repr__(self) -> str:
        preview = self.progress
        if len(preview) > 30:
            preview = preview[:27] + "..."

        return (
            f"<{TableNames.MISSION_PROGRESS.value} ("
            f"id={self.id}, "
            f"sample_id={self.sample_id}, "
            f"timestamp={self.timestamp}, "
            f"agent_id={self.agent_id}, "
            f"progress='{preview}'"
            f")>"
        )


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    engine = create_engine("sqlite://", echo=False)
    Base.metadata.create_all(engine)

    with Session(engine) as session:

        # ===== Test Case #1 : Adding samples =====

        samples_to_be_added: list[Path] = [
            Path("datasamples/0001.json"),
            Path("datasamples/0002.json"),
            Path("datasamples/0001.json"),  # try insert existing sample
        ]

        for sample_path in samples_to_be_added:
            try:
                sample_hash = SamplesTable.file_hash(sample_path)
                sample = SamplesTable(hash=sample_hash)
                session.add(sample)
                session.flush()
                logger.info(sample)
            except Exception as _:
                session.rollback()
                logger.info(
                    f"Cannot insert sample with path: {sample_path}. It already exists"
                )

        # ===== Test Case #2 : Adding Sent Message =====

        new_sample_path = Path("datasamples/0002.json")
        new_sample_hash = SamplesTable.file_hash(new_sample_path)
        sample = SamplesTable(hash=new_sample_hash)
        session.add(sample)
        session.flush()

        agent1 = AgentTable(
            sample=sample, agent_no=1, role=AgentRoles.SCOUT, mission="map"
        )
        agent2 = AgentTable(
            sample=sample, agent_no=2, role=AgentRoles.COMMANDER, mission="coordinate"
        )

        session.add_all([agent1, agent2])
        session.flush()

        sent_message = Messages(
            sample_id=sample.id,
            timestamp=123,
            sender=agent1,
            receiver=agent2,
            message="Hello from agent 1",
            type=MessageT.INFO,
        )

        session.add(sent_message)
        session.flush()

        msg = Messages(
            sample_id=sample.id,
            timestamp=123,
            sender=agent1,
            receiver=agent2,
            message="Hello Again",
            type=MessageT.INFO,
        )

        session.add(msg)
        session.flush()
