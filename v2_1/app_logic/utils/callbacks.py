from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional, Tuple

from v2_1.app_logic.data_models.agent import Agent


@dataclass
class CallbackResponse:
    """Response object from a callback function"""

    success: bool
    message: str
    status_code: int = 200
    data: Optional[Dict[str, Any]] = None


def update_information_about_agents_callback(
    agent: Agent, other_agents: list[Agent]
) -> CallbackResponse:
    """Update information about agnents for specific agent"""
    new_info = {}
    for other_agent in other_agents:
        new_info[
            other_agent.agent_id
        ] = other_agent.get_current_information_about_agent()
    agent.update_information_about_agents(new_info)
    return CallbackResponse(
        success=True,
        message=f"Information about agents updated for agent {agent.agent_id}",
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
