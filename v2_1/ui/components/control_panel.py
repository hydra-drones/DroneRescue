import streamlit as st

from v2_1.app_logic.services.session_state_controller import SceneController
from v2_1.app_logic.utils.callbacks import send_message_to_agent
from v2_1.app_logic.utils.common import execute_callback


def create_messaging_ui(agent_id: int, controller: SceneController):
    """Create a messaging UI for sending messages between agents."""
    st.markdown("### Send message to agent")
    col1, col2 = st.columns([1, 2], gap="medium")

    # Message content area
    with col2:
        st.text_area(
            "Enter your message", key=f"message_from_agent_{agent_id}", height=150
        )

    # Controls area
    with col1:
        # Message type selector
        st.selectbox(
            "Message type",
            options=("info", "order"),
            key=f"send_msg_type_from_agent_{agent_id}",
        )

        # Recipient selector
        st.selectbox(
            "Agent ID",
            options=(
                [
                    agent_id_option
                    for agent_id_option in list(controller.sampled_agents.keys())
                    if agent_id_option != agent_id
                ]
            ),
            key=f"selected_agent_for_agent_{agent_id}",
        )

        # Confirmation checkbox
        st.checkbox(
            "Send message",
            key=f"confirm_sending_msg_from_agent_{agent_id}",
            value=False,
        )

        # Send button
        st.button(
            "Send",
            on_click=execute_callback,
            args=(
                send_message_to_agent,
                controller.sampled_agents[agent_id],
                controller.sampled_agents[
                    st.session_state[f"selected_agent_for_agent_{agent_id}"]
                ],
                st.session_state[f"confirm_sending_msg_from_agent_{agent_id}"],
                st.session_state[f"message_from_agent_{agent_id}"],
                st.session_state[f"send_msg_type_from_agent_{agent_id}"],
                10,  # placeholder for actual timestamp
            ),
            key=f"send_msg_from_agent_{agent_id}",
        )
