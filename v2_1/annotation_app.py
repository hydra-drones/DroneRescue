from v2_1.session_state_controller import SceneController
import streamlit as st
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
from omegaconf import OmegaConf
from datetime import datetime
import streamlit.components.v1 as components

st.set_page_config(layout="wide")
st.title("Drone Rescue. Annotation App")

# CANCELLED: Notify before reloading page with F5
# DONE: Add scaling for scene rendering
# TODO: Enable moving left, right and down
# TODO: Reorganize UX

scene_col, col1, col2, col3 = st.columns(4, border=True)


def get_config():
    if not GlobalHydra.instance().is_initialized():
        initialize(config_path="./")
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


def move_dot(direction):
    if direction == "up":
        st.session_state.scene = st.session_state.controller.go_up(
            st.session_state.instance_type,
            st.session_state.active_agent_id,
            st.session_state.step,
        )
    # elif direction == "down" and st.session_state.dot_position[1] < max_y:
    #     st.session_state.dot_position[1] += step
    # elif direction == "left" and st.session_state.dot_position[0] > 0:
    #     st.session_state.dot_position[0] -= step
    # elif direction == "right" and st.session_state.dot_position[0] < max_x:
    #     st.session_state.dot_position[0] += step


# Control Panel
with col1:
    st.write("Control Panel")
    col11, col12 = st.columns(2)
    with col11:
        st.selectbox(
            "Agent instance", options=("agent", "target", "base"), key="instance_type"
        )
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
        st.number_input("Go with step", value=5, key="step", min_value=1)

# Scene Column
with scene_col:
    st.markdown(st.session_state.scene, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Left", on_click=move_dot, args=("left",))
    with col2:
        st.button("Up", on_click=move_dot, args=("up",))
        st.button("Down", on_click=move_dot, args=("down",))
    with col3:
        st.button("Right", on_click=move_dot, args=("right",))
