from collections import defaultdict
from datetime import datetime
from operator import sub
from typing import Dict, Literal, Optional, Tuple
import numpy as np
import streamlit as st

from v2_1.generate_sample import DatasetGenerator, AgentData, TargetData, BaseData
from v2_1.app_utils.render_objects_to_html import (
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
        self.sampled_agents: Optional[Dict[int, AgentData]] = None
        self.sampled_targets: Optional[Dict[int, TargetData]] = None
        self.sampled_bases: Optional[Dict[int, BaseData]] = None
        self.rendered_scene = None

    def sample_instances(
        self,
    ) -> Tuple[Dict[int, AgentData], Dict[int, TargetData], Dict[int, BaseData]]:
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
        widget_style = f"""
            position: relative;
            width:  {self.scene_metadata.get("size_of_mission_area")[0]}px;
            height: {self.scene_metadata.get("size_of_mission_area")[1]}px;
            border: 1px solid black;
            """

        rendered_scene_instances = (
            render_agent(self.sampled_agents, self.cfg["agents"])
            + render_target(self.sampled_targets, self.cfg["targets"])
            + render_base(self.sampled_bases, self.cfg["base"])
        )

        self.scene = f"""
        <div style="{widget_style}"> {rendered_scene_instances} </div>
        """
        return self.scene

    def go_up(
        self,
        instance_type: Literal["agent", "target", "base"],
        instance_id: int,
        step: int = 1,
    ):

        # TODO: optimize this function to re-use same components

        if instance_type == "agent":
            if instance_id in self.sampled_agents.keys():
                current_pos = self.sampled_agents[instance_id].position
                new_pos = (current_pos[0], current_pos[1] - step)
                self.sampled_agents[instance_id].position = new_pos
                logging.info("Set %s to new position %s", instance_type, str(new_pos))
            else:
                logging.warning(
                    "You are trying to move %s with id %d. %s not found with this ID",
                    instance_type,
                    instance_id,
                    instance_type,
                )
                return self.scene

        elif instance_type == "target":
            if instance_id in self.sampled_targets.keys():
                current_pos = self.sampled_targets[instance_id].position
                new_pos = (current_pos[0], current_pos[1] - step)
                self.sampled_targets[instance_id].position = new_pos
            else:
                logging.warning(
                    "You are trying to move %s with id %d. %s not found with this ID",
                    instance_type,
                    instance_id,
                    instance_type,
                )
                return self.scene

        elif instance_type == "base":
            if instance_id in self.sampled_bases.keys():
                current_pos = self.sampled_bases[instance_id].position
                new_pos = (current_pos[0], current_pos[1] - step)
                self.sampled_bases[instance_id].position = new_pos
            else:
                logging.warning(
                    "You are trying to move %s with id %d. %s not found with this ID",
                    instance_type,
                    instance_id,
                    instance_type,
                )
                return self.scene

        else:
            logging.warning("Instance type %s not found", instance_type)

        return self.render_scene()
