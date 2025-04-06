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


def send_message_to_agent(
    sender: Agent,
    receiver: Agent,
    confirm_sending: bool,
    message: str,
    message_type: Literal["info", "order"],
    timestamp: int,
) -> CallbackResponse:
    """Send message to agent"""

    sender_id = sender.agent_id
    receiver_id = receiver.agent_id

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
        timestamp=timestamp,
        receiver_id=receiver_id,
        message=message,
        message_type=message_type,
    )

    receiver.add_message_from_agent(
        timestamp=timestamp,
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
