import pytest
from sqlalchemy.orm import sessionmaker
from src.database.db import (
    AgentTable,
    Base,
    Strategy,
    MessageT,
    StrategyT,
    PositionT,
    Messages,
    Samples,
    Positions,
)
from pathlib import Path


@pytest.fixture(scope="session")
def tables(engine):
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session(engine, tables):
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture()
def add_agent(db_session):
    new_sample_path = Path("datasamples/0002.json")
    new_sample_hash = Samples.file_hash(new_sample_path)
    sample = Samples(hash=new_sample_hash)
    db_session.add(sample)
    db_session.flush()

    agent = AgentTable(sample=sample, agent_no=1, role="explorer", mission="map")

    db_session.add(agent)
    db_session.flush()

    metadata = {"sample_id": agent.sample_id}

    return agent, metadata


@pytest.fixture()
def add_agents(db_session):
    new_sample_path = Path("datasamples/0002.json")
    new_sample_hash = Samples.file_hash(new_sample_path)
    sample = Samples(hash=new_sample_hash)
    db_session.add(sample)
    db_session.flush()

    agent1 = AgentTable(sample=sample, agent_no=1, role="explorer", mission="map")
    agent2 = AgentTable(
        sample=sample, agent_no=2, role="commander", mission="coordinate"
    )

    db_session.add_all([agent1, agent2])
    db_session.flush()

    return agent1, agent2, sample


@pytest.fixture()
def agent1_sent_message_to_agent2(add_agents):
    agent1, agent2, sample = add_agents

    msg = Messages(
        sample_id=agent1.sample_id,
        timestamp=123,
        sender=agent1,
        receiver=agent2,
        message="Hello Again",
        type=MessageT.INFO,
    )

    return agent1, agent2, sample, msg


def test_add_samples(db_session):
    samples_to_be_added: list[Path] = [
        Path("tests/assets/0001.json"),
        Path("tests/assets/0002.json"),
        Path("tests/assets/0001.json"),  # try insert existing sample
    ]

    success_inserts = 0
    failed_inserts = 0

    for sample_path in samples_to_be_added:
        sample_hash = Samples.file_hash(sample_path)
        sample = Samples(hash=sample_hash)

        try:
            db_session.add(sample)
            db_session.flush()
            success_inserts += 1
        except Exception as e:
            db_session.rollback()
            failed_inserts += 1
            print(f"Ошибка вставки: {e}")

    assert success_inserts == 2
    assert failed_inserts == 1


def test_add_sent_message(db_session, agent1_sent_message_to_agent2):

    agent1, agent2, sample, msg = agent1_sent_message_to_agent2

    db_session.add(msg)
    db_session.flush()

    agent1_sent_message_to_agent2 = agent1.messages_sent[0]
    received_msg = agent2.messages_received[0]

    assert (
        agent1_sent_message_to_agent2 == received_msg
    ), f"The both messages received and sent are different: ({agent1_sent_message_to_agent2}) / ({received_msg})"
    assert (
        agent2.messages_sent == []
    ), "In this test the second agent should only receive the message, but sent message was found too"
    assert (
        msg.sample_id == sample.id
    ), f"Sample ID: {msg.sample_id} of the message is not equal to sample ID {sample.id}"


def test_add_2_sent_message(db_session, agent1_sent_message_to_agent2):

    agent1, agent2, _, msg = agent1_sent_message_to_agent2

    db_session.add(msg)
    db_session.add(msg)
    db_session.flush()

    num_sent_messages = len(agent1.messages_sent)
    num_received_messages = len(agent2.messages_received)

    assert (
        num_sent_messages == num_received_messages
    ), f"Different number of received and sent messages: {num_sent_messages} and {num_received_messages}"


def test_changing_strategy(db_session, add_agent):
    agent, metadata = add_agent

    sample_id = metadata["sample_id"]

    strategy = Strategy(
        sample_id=agent.sample_id,
        agent=agent,
        timestamp=300,
        strategy="Test local strategy",
        type=StrategyT.GLOBAL,
    )

    db_session.add(strategy)
    db_session.flush()

    assert (
        strategy.agent.id == agent.id
    ), f"Got different agent ID for local strategy: {agent.id}, got {strategy.agent.id}"
    assert (
        strategy.sample_id == sample_id
    ), f"Got different sample_id for the local strategy: {sample_id}, got {strategy.sample_id}"
    assert strategy.type == StrategyT.GLOBAL


def test_adding_new_agent_position(db_session, add_agent):
    agent, _ = add_agent

    pos = Positions(
        sample_id=agent.sample_id,
        timestamp=100,
        agent=agent,
        pos_x=10,
        pos_y=20,
        type=PositionT.AGENT,
    )

    db_session.add(pos)
    db_session.flush()

    assert pos.sample_id == agent.sample_id
