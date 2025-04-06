from typing import Dict, Literal, Optional, Tuple

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

    def sample_instances(
        self,
    ) -> Tuple[Dict[int, Agent], Dict[int, TargetData], Dict[int, BaseData]]:
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

            if current_timestamp == self.global_timestamp:
                instances[instance_id].update_timestamp_and_set_new_position(
                    step, new_pos
                )
                self.global_timestamp += step
            elif current_timestamp + step > self.global_timestamp:
                logging.warning(
                    "After taking the step, the timestamp of %s will be greater than the global timestamp",
                )
                return self.scene
            else:
                instances[instance_id].update_timestamp_and_set_new_position(
                    step, new_pos
                )

        else:
            instances[instance_id].position = new_pos
            logging.info("Set %s to new position %s", instance_type, str(new_pos))

        return self.render_scene()
