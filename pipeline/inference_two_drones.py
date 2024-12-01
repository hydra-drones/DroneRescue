import hydra
from pathlib import Path
from omegaconf import DictConfig
import logging
import sys

import numpy as np
import json
from agent.llm_agent import LLMAgent
from environment.environment import Environment
from environment.constants import OBSTACLE_MAP, OBJECT_MAP, COLOR_MAP
from dotenv import load_dotenv

import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger("inference_script")
logger.setLevel(logging.INFO)


def step_in_environment(
    agent_num: int, env: Environment, agent: LLMAgent, **kwargs
) -> tuple[bool, LLMAgent, dict]:
    """
    Make the action in the environment
    """
    action = kwargs["action"]
    agent_position = kwargs["agent_position"]
    speed = kwargs["speed"]
    rollout_config = kwargs["rollout_config"]
    step_num = kwargs["step_num"]
    explaination = kwargs["explaination"]
    agent_trajectories_folder = kwargs["agent_trajectories_folder"]
    message_to_another_drone = kwargs["message_to_another_drone"]
    observaion_area = agent.observation_area

    (
        new_agent_position,
        done,
        terminated,
        observation,
        obs_binary_mask,
        env_metadata,
    ) = env.step(action, agent_position, observaion_area, speed)

    agent.update_visited_map(observation, obs_binary_mask)

    # Add current agent position to the visited map
    visited_map_with_cur_agent_position = agent.visited_map.copy()
    visited_map_with_cur_agent_position[new_agent_position] = 4

    # Add current agent position to the local observation
    observation_with_cur_agent_position = observation.copy()
    observation_with_cur_agent_position[
        env_metadata["agent_position_in_local_observation"]
    ] = 4

    visited_map_name = f"visited_map_{step_num}.jpg"
    rollout_config[agent_num][step_num] = {
        "metadata": {
            "action": action,
            "speed": speed,
            "explaination": explaination,
            "message_to_another_drone": message_to_another_drone,
            "done": done,
            "terminated": terminated,
        },
        "visited_map_path": str(agent_trajectories_folder / visited_map_name),
    }

    env.render_visited_map(
        visited_map_with_cur_agent_position,
        str(agent_trajectories_folder / visited_map_name),
    )

    return (
        terminated,
        agent,
        visited_map_with_cur_agent_position,
        observation_with_cur_agent_position,
        new_agent_position,
        rollout_config,
    )


@hydra.main(version_base=None, config_path="configs", config_name="inference_config")
def run_inference(cfg: DictConfig):
    rollout_config = {1: {}, 2: {}}

    # Stage 1 : Prepare directory
    rollout_folder = Path(cfg.structure.exp_name)
    if rollout_folder.exists():
        versions = rollout_folder.rglob("version_*")
        versions = [int(version.name.replace("version_", "")) for version in versions]
        if versions:
            last_version = max(versions) + 1
        else:
            last_version = 1

        rollout_folder = rollout_folder / f"version_{last_version}"
    else:
        rollout_folder = rollout_folder / f"version_1"
        rollout_folder.mkdir(parents=True)

    agent_01_rollout_folder = rollout_folder / "agent_01"
    agent_01_trajectories_folder = agent_01_rollout_folder / "trajectories"
    agent_01_trajectories_folder.mkdir(parents=True, exist_ok=True)

    agent_02_rollout_folder = rollout_folder / "agent_02"
    agent_02_trajectories_folder = agent_02_rollout_folder / "trajectories"
    agent_02_trajectories_folder.mkdir(parents=True, exist_ok=True)

    # Stage 2 : Initialize agents
    agent_01 = LLMAgent(
        area_size=tuple(cfg.environment.area_size),
        object_map=OBJECT_MAP,
        observation_area=tuple(cfg.agents.agent_01.observation_area),
        api_key=api_key,
        chat_gpt_model="gpt-4o-2024-08-06",
    )

    agent_02 = LLMAgent(
        area_size=tuple(cfg.environment.area_size),
        object_map=OBJECT_MAP,
        observation_area=tuple(cfg.agents.agent_02.observation_area),
        api_key=api_key,
        chat_gpt_model="gpt-4o-2024-08-06",
    )

    # Stage 3 : Initialize environment
    env = Environment(
        obstacles_map=OBSTACLE_MAP,
        color_map=COLOR_MAP,
        object_map=OBJECT_MAP,
        area_size=tuple(cfg.environment.area_size),
        num_targets=cfg.environment.targets_num,
        num_obstacles=cfg.environment.obstacles_num,
    )

    env_rendered_path = rollout_folder / "environment.jpg"
    env.render_env(str(env_rendered_path))
    rollout_config["env"] = str(env_rendered_path)

    # Stage 4 : Initialize starting positions
    agent_01_position = env.generare_start_positions((10, 1))
    agent_02_position = env.generare_start_positions((10, 2))

    initialization_action = 4

    (
        agent_01_done_flag,
        agent_01,
        agent_01_vis_map,
        agent_01_observation,
        agent_01_new_position,
        rollout_config,
    ) = step_in_environment(
        1,
        env,
        agent_01,
        action=initialization_action,
        agent_position=agent_01_position,
        speed=1,
        rollout_config=rollout_config,
        step_num=0,
        explaination="",
        message_to_another_drone="",
        agent_trajectories_folder=agent_01_trajectories_folder,
    )

    (
        agent_02_done_flag,
        agent_02,
        agent_02_vis_map,
        agent_02_observation,
        agent_02_new_position,
        rollout_config,
    ) = step_in_environment(
        1,
        env,
        agent_02,
        action=initialization_action,
        agent_position=agent_02_position,
        speed=1,
        rollout_config=rollout_config,
        step_num=0,
        explaination="",
        message_to_another_drone="",
        agent_trajectories_folder=agent_02_trajectories_folder,
    )

    agent_01_communcation_message = ""
    agent_02_communcation_message = ""

    step = 0
    while True:
        step += 1
        agent_01_response = agent_01.generate_action_by_model(
            cfg.environment.targets_num,
            agent_01_vis_map,
            agent_01_observation,
            agent_02_communcation_message,
        )

        agent_01_communcation_message = agent_01_response["message_to_agent"]

        (
            agent_01_done_flag,
            agent_01,
            agent_01_vis_map,
            agent_01_observation,
            agent_01_new_position,
            rollout_config,
        ) = step_in_environment(
            1,
            env,
            agent_01,
            action=agent_01_response["action"],
            agent_position=agent_01_new_position,
            speed=agent_01_response["speed"],
            rollout_config=rollout_config,
            step_num=step,
            explaination=agent_01_response["explaination"],
            message_to_another_drone=agent_01_response["message_to_agent"],
            agent_trajectories_folder=agent_01_trajectories_folder,
        )

        logger.info(
            f"""\nAgent (1) :\nStep_{step}:
            A: {agent_01_response["action"]}
            S: {agent_01_response["speed"]}.
            Explainataion: {agent_01_response["explaination"]}\n"""
        )

        agent_02_response = agent_02.generate_action_by_model(
            cfg.environment.targets_num,
            agent_02_vis_map,
            agent_02_observation,
            agent_01_communcation_message,
        )

        agent_02_communcation_message = agent_01_response["message_to_agent"]

        (
            agent_02_done_flag,
            agent_02,
            agent_02_vis_map,
            agent_02_observation,
            agent_02_new_position,
            rollout_config,
        ) = step_in_environment(
            2,
            env,
            agent_02,
            action=agent_02_response["action"],
            agent_position=agent_02_new_position,
            speed=agent_02_response["speed"],
            rollout_config=rollout_config,
            step_num=step,
            explaination=agent_02_response["explaination"],
            message_to_another_drone=agent_02_response["message_to_agent"],
            agent_trajectories_folder=agent_02_trajectories_folder,
        )

        logger.info(
            f"""\nAgent (2) :\nStep_{step}:
            A: {agent_02_response["action"]}
            S: {agent_02_response["speed"]}.
            Explainataion: {agent_02_response["explaination"]}\n"""
        )

        logger.info(f"Agent 01 message : {agent_01_communcation_message}")

        logger.info(f"Agent 02 message : {agent_02_communcation_message}")

        with open(str(rollout_folder / "rollout.json"), "w") as file:
            json.dump(rollout_config, file)

        if agent_01_response["mission_completed"]:
            sys.exit("Agent (1) sent the request to finish the mission")

        if agent_02_response["mission_completed"]:
            sys.exit("Agent (2) sent the request to finish the mission")

        # if agent_01_done_flag:
        #     sys.exit("Script stopped due to terminated of agent (1)")

        # if agent_02_done_flag:
        #     sys.exit("Script stopped due to terminated of agent (2)")

        # if (agent_01.targets_found + agent_02.targets_found) == cfg.environment.targets_num:
        #     sys.exit("All targets was found!")


if __name__ == "__main__":
    run_inference()
