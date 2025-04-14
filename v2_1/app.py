from v2_1.app_logic.services.session_state_controller import SceneController
from v2_1.ui.components.control_panel import (
    create_messaging_ui,
    update_global_strategy_ui,
    update_info_about_agents_ui,
    update_local_strategy_ui,
)
from v2_1.app_logic.data_models.agent import Agent
import streamlit as st
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import OmegaConf

# DONE: Allow to increase the agent timestamp up to global timsestamp without perfoming action
# TODO: Save information about target position and base position
# CANCELLED: Save the scene if only all agents are in the same timstamp
# TODO: Add agent's mission

st.set_page_config(layout="wide")

scene_col, col1 = st.columns([1, 2], gap="medium")


def get_config():
    if not GlobalHydra.instance().is_initialized():
        initialize(config_path="./config")
    cfg = compose(config_name="setup")
    return OmegaConf.to_container(cfg, resolve=True)


if "controller" not in st.session_state:
    config = get_config()
    controller = SceneController(config)
    st.session_state.controller = controller

if ("is_instances_sampled" not in st.session_state) and (
    "controller" in st.session_state
):
    st.session_state.controller.sample_instances()
    st.session_state.is_instances_sampled = True

if "scene" not in st.session_state:
    st.session_state.scene = st.session_state.controller.render_scene()


def move_instance(direction):
    st.session_state.scene = st.session_state.controller.move(
        st.session_state.instance_type,
        st.session_state.active_agent_id,
        direction,
        st.session_state.step,
    )


# Control Panel
with col1:
    st.text(f"Global timestamp: {st.session_state.controller.global_timestamp}")
    for agent_id in st.session_state.controller.sampled_agents:
        agent: Agent = st.session_state.controller.sampled_agents[agent_id]
        with st.expander(
            f"{agent.role} {agent_id} : {agent.position} ||| {agent.get_latest_timestamp()}"
        ):
            st.button(
                "Increase timestamp to global",
                key=f"increase_timestamp_to_global_for_agent_{agent_id}",
                on_click=st.session_state.controller.increase_local_timestamp_to_global,
                args=(agent_id,),
            )
            create_messaging_ui(
                agent_id,
                st.session_state.controller,
            )
            update_info_about_agents_ui(
                agent_id,
                st.session_state.controller,
            )
            update_global_strategy_ui(
                agent_id,
                st.session_state.controller,
            )
            update_local_strategy_ui(
                agent_id,
                st.session_state.controller,
            )


# Scene Column
with scene_col:
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Current datasample ID: {st.session_state.controller.datasample_id}")
    with col2:
        st.button(
            "Save sampled scene", on_click=st.session_state.controller.save_datasample
        )

    st.markdown(st.session_state.scene, unsafe_allow_html=True)
    st.checkbox(
        "Edit Mode",
        value=False,
        key="edit_mode",
        on_change=lambda: st.session_state.controller.set_edit_mode(
            st.session_state.edit_mode
        ),
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.selectbox(
            "Agent instance", options=("agent", "target", "base"), key="instance_type"
        )
        st.button("Left", on_click=move_instance, args=("left",))
    with col2:
        st.number_input(
            "Agent ID",
            value=1,
            key="active_agent_id",
            min_value=1,
            max_value=(
                len(st.session_state.controller.sampled_agents)
                if st.session_state.instance_type == "agent"
                else len(st.session_state.controller.sampled_targets)
                if st.session_state.instance_type == "target"
                else len(st.session_state.controller.sampled_bases)
            ),
        )
        st.button("Up", on_click=move_instance, args=("up",))
        st.button("Down", on_click=move_instance, args=("down",))
    with col3:
        st.number_input("Go with step", value=5, key="step", min_value=1)
        st.button("Right", on_click=move_instance, args=("right",))
