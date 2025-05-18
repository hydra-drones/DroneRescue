import streamlit as st

from src.annotation_app.app_logic.services.session_state_controller import (
    SceneController,
)
from src.annotation_app.app_logic.utils.common import execute_callback


def update_global_strategy_ui(agent_id: int, controller: SceneController):
    """Update the global strategy for the agent."""
    st.markdown("**Update global strategy**")
    st.text_area(
        label="Enter your message",
        label_visibility="collapsed",
        key=f"global_strategy_for_agent_{agent_id}",
    )
    if st.button("Update", key=f"update_global_strategy_for_agent_{agent_id}"):
        execute_callback(
            controller.update_gloabl_strategy_for_agent_callback,
            agent_id,
            st.session_state[f"global_strategy_for_agent_{agent_id}"],
        )


def update_local_strategy_ui(agent_id: int, controller: SceneController):
    """Update the local strategy for the agent."""
    st.markdown("**Update local strategy**")
    st.text_area(
        "Enter your message",
        label_visibility="collapsed",
        key=f"local_strategy_for_agent_{agent_id}",
    )
    if st.button("Update", key=f"update_local_strategy_for_agent_{agent_id}"):
        execute_callback(
            controller.update_local_strategy_for_agent_callback,
            agent_id,
            st.session_state[f"local_strategy_for_agent_{agent_id}"],
        )


def update_mission_progress_ui(agent_id: int, controller: SceneController):
    """Update the local strategy for the agent."""
    st.markdown("**Update mission progress**")
    st.text_area(
        "Enter your message",
        label_visibility="collapsed",
        key=f"mission_progress_{agent_id}",
    )
    if st.button("Update", key=f"mission_progress_for_agent_{agent_id}"):
        execute_callback(
            controller.update_mission_progress_callback,
            agent_id,
            st.session_state[f"mission_progress_{agent_id}"],
        )


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

        # Get all other agents
        other_agents = [
            agent_id_option
            for agent_id_option in list(controller.sampled_agents.keys())
            if agent_id_option != agent_id
        ]

        # Create checkboxes for each agent
        for other_agent_id in other_agents:
            agent_role = controller.sampled_agents[other_agent_id].role
            checkbox_key = f"recipient_checkbox_{agent_id}_{other_agent_id}"

            st.checkbox(
                f"{agent_role} {other_agent_id}",
                key=checkbox_key,
            )

        # Confirmation checkbox
        st.checkbox(
            "Send message",
            key=f"confirm_sending_msg_from_agent_{agent_id}",
            value=False,
        )

        # Send button
        if st.button(
            "Send",
            key=f"send_msg_from_agent_{agent_id}",
        ):
            # Get the selected recipients
            recipients = [
                recipient_id
                for recipient_id in other_agents
                if st.session_state[f"recipient_checkbox_{agent_id}_{recipient_id}"]
            ]

            if not recipients:
                st.warning("Please select at least one recipient")
            else:
                # Send message to each selected recipient
                for recipient_id in recipients:
                    execute_callback(
                        controller.send_message_to_agent,
                        agent_id,
                        recipient_id,
                        st.session_state[f"confirm_sending_msg_from_agent_{agent_id}"],
                        st.session_state[f"message_from_agent_{agent_id}"],
                        st.session_state[f"send_msg_type_from_agent_{agent_id}"],
                    )
