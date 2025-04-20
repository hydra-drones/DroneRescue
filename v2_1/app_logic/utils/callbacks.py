from typing import Literal

from v2_1.app_logic.data_models.agent import Agent
from v2_1.app_logic.services.session_state_controller import CallbackResponse


def update_gloabl_strategy_for_agent_callback(agent: Agent, new_global_strategy: str):
    """Update global strategy for agent"""
    agent.update_global_strategy(new_global_strategy)
    return CallbackResponse(
        success=True,
        message=f"Global strategy updated for agent {agent.agent_id}",
        status_code=200,
    )


def update_local_strategy_for_agent_callback(agent: Agent, new_local_strategy: str):
    """Update local strategy for agent"""
    agent.update_local_strategy(new_local_strategy)
    return CallbackResponse(
        success=True,
        message=f"Local strategy updated for agent {agent.agent_id}",
        status_code=200,
    )


def update_mission_progress_callback(agent: Agent, new_mission_progress: str):
    """Update mission progress for agent"""
    agent.update_mission_progress(new_mission_progress)
    return CallbackResponse(
        success=True,
        message=f"Mission progress updated for agent {agent.agent_id}",
        status_code=200,
    )


def send_message_to_agent(
    sender: Agent,
    receiver: Agent,
    confirm_sending: bool,
    message: str,
    message_type: Literal["info", "order"],
    global_timestamp: int,
) -> CallbackResponse:
    """Send message to agent"""

    sender_id = sender.agent_id
    receiver_id = receiver.agent_id

    sender_timestamp = sender.get_latest_timestamp()
    receiver_timestamp = receiver.get_latest_timestamp()

    if sender_timestamp != global_timestamp or receiver_timestamp != global_timestamp:
        return CallbackResponse(
            success=False,
            message=(
                f"Timestamp mismatch: sender timestamp ({sender_timestamp}), "
                f"receiver timestamp ({receiver_timestamp}), global timestamp ({global_timestamp})"
            ),
            status_code=400,
        )

    if not confirm_sending:
        return CallbackResponse(
            success=False,
            message=f"Failed to send the message from agent {sender_id} to agent {receiver_id}\nPlease confirm sending the message",
            status_code=400,
        )

    if not message:
        return CallbackResponse(
            success=False,
            message=f"Failed to send the message from agent {sender_id} to agent {receiver_id}\nPlease provide a message",
            status_code=400,
        )

    sender.add_sended_message(
        global_timestamp=global_timestamp,
        receiver_id=receiver_id,
        message=message,
        message_type=message_type,
    )

    receiver.add_message_from_agent(
        global_timestamp=global_timestamp,
        sender_id=sender_id,
        message=message,
        message_type=message_type,
    )

    return CallbackResponse(
        success=True,
        message=f"Message sent successfully from agent {sender_id} to agent {receiver_id}",
        status_code=200,
        data={
            "sender": sender_id,
            "receiver": receiver_id,
            "timestamp": "2025-04-05T10:30:00Z",
        },
    )
