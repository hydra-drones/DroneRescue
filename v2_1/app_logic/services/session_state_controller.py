from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import json

from v2_1.scripts.generate_sample import DatasetGenerator, TargetData, BaseData
from v2_1.app_logic.data_models.agent import Agent

from v2_1.ui.render.render_objects_to_html import (
    render_agent,
    render_base,
    render_target,
)
import logging

logging.basicConfig(level=logging.INFO)


class SceneController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scene_metadata = cfg["metadata"]
        self.sampled_agents: Optional[Dict[int, Agent]] = None
        self.sampled_targets: Optional[Dict[int, TargetData]] = None
        self.sampled_bases: Optional[Dict[int, BaseData]] = None
        self.rendered_scene = None
        self.global_timestamp: int = 0
        self.edit_mode: bool = False
        self._path_to_save = Path(self.scene_metadata.get("save_datasample_to"))
        self.datasample_id: int = self._get_datasample_id()

    def sample_instances(
        self,
    ) -> Tuple[Dict[int, Agent], Dict[int, TargetData], Dict[int, BaseData]]:
        """Randomly sample agents, targets, and bases from the configuration"""
        generator = DatasetGenerator(
            self.cfg["agents"],
            self.cfg["targets"],
            self.cfg["base"],
            self.cfg["metadata"],
        )

        (
            self.sampled_agents,
            self.sampled_targets,
            self.sampled_bases,
        ) = generator.sample()

        return self.sampled_agents, self.sampled_targets, self.sampled_bases

    def render_scene(self) -> str:
        """Return the HTML code for the scene"""

        scale_factor = self.scene_metadata.get("scale_factor")

        widget_style = f"""
            position: relative;
            width:  {self.scene_metadata.get("size_of_mission_area")[0] * scale_factor}px;
            height: {self.scene_metadata.get("size_of_mission_area")[1] * scale_factor}px;
            border: 1px solid black;
            """

        rendered_scene_instances = (
            render_agent(self.sampled_agents, self.cfg["agents"], scale_factor)
            + render_target(self.sampled_targets, self.cfg["targets"], scale_factor)
            + render_base(self.sampled_bases, self.cfg["base"], scale_factor)
        )

        self.scene = f"""
        <div style="{widget_style}"> {rendered_scene_instances} </div>
        """
        return self.scene

    def move(
        self,
        instance_type: Literal["agent", "target", "base"],
        instance_id: int,
        direction: Literal["up", "down"],
        step: int = 1,
    ):
        """Move the instance in the specified direction by the given step size"""

        instance_dict = {
            "agent": self.sampled_agents,
            "target": self.sampled_targets,
            "base": self.sampled_bases,
        }

        if instance_type not in instance_dict:
            logging.warning("Instance type %s not found", instance_type)
            return self.scene

        instances = instance_dict[instance_type]

        if instance_id not in instances:
            logging.warning(
                "You are trying to move %s with id %d. %s not found with this ID",
                instance_type,
                instance_id,
                instance_type,
            )
            return self.scene

        current_pos = instances[instance_id].position
        x, y = current_pos[0], current_pos[1]

        if direction == "up":
            critical_value = 0
            y = max(critical_value, (y - step))
            if y == critical_value:
                logging.warning(
                    "%s is out of the range. Stay in place", instance_type.capitalize()
                )

        elif direction == "down":
            critical_value = self.scene_metadata.get("size_of_mission_area")[1]
            y = min(critical_value, (y + step))
            if y == critical_value:
                logging.warning(
                    "%s is out of the range. Stay in place", instance_type.capitalize()
                )

        elif direction == "right":
            critical_value = self.scene_metadata.get("size_of_mission_area")[0]
            x = min(critical_value, (x + step))
            if x == critical_value:
                logging.warning(
                    "%s is out of the range. Stay in place", instance_type.capitalize()
                )

        elif direction == "left":
            critical_value = 0
            x = max(critical_value, (x - step))
            if x == critical_value:
                logging.warning(
                    "%s is out of the range. Stay in place", instance_type.capitalize()
                )

        new_pos = (x, y)

        if isinstance(instances[instance_id], Agent):
            current_timestamp = instances[instance_id].get_latest_timestamp()

            if self.edit_mode:
                instances[instance_id].update_position_in_edit_mode(new_pos)
            else:
                if current_timestamp == self.global_timestamp:
                    self._move_agent(instances[instance_id], step, new_pos)
                    self.global_timestamp += step
                elif current_timestamp + step > self.global_timestamp:
                    logging.warning(
                        "After taking the step, the timestamp of %s will be greater than the global timestamp",
                    )
                    return self.scene
                else:
                    self._move_agent(instances[instance_id], step, new_pos)
        else:
            instances[instance_id].position = new_pos
            logging.info("Set %s to new position %s", instance_type, str(new_pos))

        return self.render_scene()

    def _move_agent(self, agent: Agent, step: int, new_pos: tuple[int, int]):
        agent.update_timestamp_and_set_new_position(step, new_pos)
        #  Add targets if are visiable in the field of view
        targets_in_fov = agent.get_visiable_targets_in_fov(
            [target.position for _, target in self.sampled_targets.items()]
        )
        if targets_in_fov:
            agent.update_target_in_fov(targets_in_fov)

    def increase_local_timestamp_to_global_for_all_agents(self):
        """Increase the local timestamp of the agent to the global timestamp"""
        for agent in self.sampled_agents.values():
            agent.increase_local_timestamp_to_global_and_sync_position(
                self.global_timestamp
            )

    def freeze_targets(self) -> dict:
        """Convert the target position into dict"""
        freezed_targets = {}
        for target_id, target_data_cls in self.sampled_targets.items():
            freezed_targets[target_id] = {"position": target_data_cls.position}
        return freezed_targets

    def freeze_bases(self) -> dict:
        """Convert the base position into dict"""
        freezed_bases = {}
        for base_id, base_data_cls in self.sampled_bases.items():
            freezed_bases[base_id] = {"position": base_data_cls.position}
        return freezed_bases

    def freeze_agents(self) -> dict:
        """Save all agents' states into the dict"""
        freezed_agents = {}
        for agent_id, agent_cls in self.sampled_agents.items():
            freezed_agents[agent_id] = agent_cls.freeze_agent_state()
        return freezed_agents

    def get_freezed_scene(self) -> dict:
        """Collect all the scene data into the dict"""
        freezed_scene = {
            "agents": self.freeze_agents(),
            "targets": self.freeze_targets(),
            "bases": self.freeze_bases(),
            "area": self.scene_metadata.get("size_of_mission_area"),
        }
        return freezed_scene

    def save_datasample(self):
        """Save scene data into json file"""
        if not self._path_to_save.exists():
            self._path_to_save.mkdir(parents=True, exist_ok=True)

        filename = f"{self._path_to_save}/{self.datasample_id}.json"

        with open(filename, "w", encoding="utf-8") as f:
            # Convert the dictionary to a JSON string
            json_str = json.dumps(self.get_freezed_scene(), indent=4)
            f.write(json_str)
            logging.info("Scene saved to %s", filename)
        return filename

    def set_edit_mode(self, edit_mode: bool):
        """Set edit mode"""
        self.edit_mode = edit_mode
        logging.info("Edit mode set to %s", edit_mode)

    def _get_datasample_id(self) -> int:
        if not self._path_to_save.exists():
            self._path_to_save.mkdir(parents=True, exist_ok=True)

        last_id = self._get_last_id_in_datadir()
        formatted_id = f"{last_id:04d}"

        return formatted_id

    def _get_last_id_in_datadir(self) -> int:
        files = list(self._path_to_save.glob("*.json"))
        if not files:
            return 0
        last_file = max(files, key=lambda x: int(x.stem))
        return int(last_file.stem) + 1
