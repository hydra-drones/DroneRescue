import hydra
from pathlib import Path
from omegaconf import DictConfig
import logging

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


@hydra.main(version_base=None, config_path="configs", config_name="inference_config")
def run_inference(cfg: DictConfig):

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

    # Stage 2 : Initialize agents
    agent = LLMAgent(
        area_size=tuple(cfg.environment.area_size),
        object_map=OBJECT_MAP,
        observation_area=tuple(cfg.agents.agent_01.observation_area),
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

    # Stage 4 : Initialize starting positions
    agent_position = env.generare_start_positions((10, 1))

    agent_position, done, terminated, observation, obs_binary_mask, metadata = env.step(
        4, agent_position, agent.observation_area, 1
    )

    # Stage 5 : Update visited map
    agent.update_visited_map(observation, obs_binary_mask)

    # Stage 6 : Add current agent position to the visited map
    visited_map_with_cur_agent_position = agent.visited_map.copy()
    visited_map_with_cur_agent_position[agent_position] = 4

    # Stage 7 : Add current agent position to the local observation
    observation_with_cur_agent_position = observation.copy()
    observation_with_cur_agent_position[
        metadata["agent_position_in_local_observation"]
    ] = 4

    rollout_config = {}
    for step in range(cfg.rollout_num):

        if done:
            logging.info("Win!")

        if terminated:
            logging.info("Fail")

        response = agent.generate_action_by_model(
            visited_map_with_cur_agent_position, observation_with_cur_agent_position
        )
        action = response["action"]
        explaination = response["explaination"]
        speed = response["speed"]
        logger.info(
            f"Step_{step}: A: {action} S: {speed}. Explainataion: {explaination}"
        )

        (
            agent_position,
            done,
            terminated,
            observation,
            obs_binary_mask,
            metadata,
        ) = env.step(action, agent_position, agent.observation_area, speed)

        # Update visited map
        agent.update_visited_map(observation, obs_binary_mask)

        # Add current agent position to the visited map
        visited_map_with_cur_agent_position = agent.visited_map.copy()
        visited_map_with_cur_agent_position[agent_position] = 4

        # Add current agent position to the local observation
        observation_with_cur_agent_position = observation.copy()
        observation_with_cur_agent_position[
            metadata["agent_position_in_local_observation"]
        ] = 4

        agent_rollout_folder = rollout_folder / "agent_01"

        agent_trajectories_folder = agent_rollout_folder / "trajectories"
        agent_trajectories_folder.mkdir(parents=True, exist_ok=True)
        visited_map_name = f"visited_map_{step}.jpg"

        env.render_visited_map(
            visited_map_with_cur_agent_position,
            str(agent_trajectories_folder / visited_map_name),
        )

        rollout_config[step] = {
            "metadata": {
                "action": action,
                "speed": speed,
                "explaination": explaination,
            },
            "visited_map_path": str(agent_trajectories_folder / visited_map_name),
        }

    env_rendered_path = agent_rollout_folder / "environment.jpg"
    env.render_env(str(env_rendered_path))
    rollout_config["env"] = str(env_rendered_path)

    with open(str(agent_rollout_folder / "rollout.json"), "w") as file:
        json.dump(rollout_config, file)


if __name__ == "__main__":
    run_inference()
