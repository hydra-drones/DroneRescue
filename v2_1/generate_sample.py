import numpy as np
import hydra
from omegaconf import DictConfig
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)


@dataclass
class AgentData:
    role: str
    position: Tuple[int, int]


@dataclass
class TargetData:
    position: Tuple[int, int]


@dataclass
class BaseData:
    position: Tuple[int, int]


class DatasetGenerator:
    def __init__(self, agents, targets, base, metadata):
        self.agents = agents
        self.targets = targets
        self.base = base
        self.metadata = metadata
        self.sampled_agents: Optional[Dict[int, AgentData]] = None
        self.sampled_targets: Optional[Dict[int, TargetData]] = None
        self.sampled_bases: Optional[Dict[int, BaseData]] = None

    def sample(
        self,
    ) -> Tuple[Dict[int, AgentData], Dict[int, TargetData], Dict[int, BaseData]]:
        self._sample_agents()
        self._sample_targets()
        self._sample_base()

        return self.sampled_agents, self.sampled_targets, self.sampled_bases

    def _sample_agents(self):
        self.sampled_agents = defaultdict()
        _idx = 1

        for agent in self.agents.values():
            number_of_agents = np.random.randint(
                agent["number"]["min"], agent["number"]["max"] + 1
            )

            for _ in range(number_of_agents):
                position = self._generate_instance_position_in_2D_space(
                    agent["position_range"]["min_yx"],
                    agent["position_range"]["max_yx"],
                )

                self.sampled_agents[_idx] = AgentData(
                    role=agent["role"], position=position
                )

                _idx += 1

    def _sample_targets(self):
        self.sampled_targets = defaultdict()
        _idx = 1

        number_of_targets = np.random.randint(
            self.targets["number"]["min"], self.targets["number"]["max"] + 1
        )
        for _ in range(number_of_targets):
            position = self._generate_instance_position_in_2D_space(
                self.targets["position_range"]["min_yx"],
                self.targets["position_range"]["max_yx"],
            )

            self.sampled_targets[_idx] = TargetData(position=position)

            _idx += 1

    def _sample_base(self):
        self.sampled_bases = defaultdict()
        _idx = 1

        base_position_range_idx = np.random.randint(
            len(self.base["possible_position_ranges"])
        )
        base_position_range = self.base["possible_position_ranges"][
            base_position_range_idx
        ]

        base_position_x = (
            base_position_range[0][1]
            if base_position_range[0][1] == base_position_range[1][1]
            else np.random.randint(
                base_position_range[0][1], base_position_range[1][1] + 1
            )
        )
        base_position_y = (
            base_position_range[0][0]
            if base_position_range[0][0] == base_position_range[1][0]
            else np.random.randint(
                base_position_range[0][0], base_position_range[1][0] + 1
            )
        )

        self.sampled_bases[_idx] = BaseData(position=[base_position_x, base_position_y])

    def _generate_instance_position_in_2D_space(self, min_yx, max_yx):
        y_min, x_min = min_yx
        y_max, x_max = max_yx
        return np.random.randint(y_min, y_max + 1), np.random.randint(x_min, x_max + 1)

    # TODO: update code for visualization on plot

    # def visualize(self):
    #     self.plot = self._generate_base_plot()

    #     self._add_agents_to_plot()
    #     self._add_targets_to_plot()
    #     self._add_base_to_plot()

    #     self.plot.savefig("sample.png")
    #     self.plot.close()

    # def _generate_base_plot(self):
    #     plt.figure(figsize=(6, 6))
    #     plt.xlim(0, self.metadata["size_of_mission_area"][1])
    #     plt.ylim(0, self.metadata["size_of_mission_area"][1])
    #     plt.gca().invert_yaxis()
    #     plt.gca().xaxis.set_ticks_position("top")
    #     plt.grid(True)
    #     plt.xlabel("X")
    #     plt.ylabel("Y")

    #     return plt

    # def _add_agents_to_plot(self):
    #     for agent in self.agents.values():
    #         for agent_id, position in enumerate(agent["positions"]):
    #             self.plot.scatter(
    #                 position[1],
    #                 position[0],
    #                 marker=agent["symbol"],
    #                 color=agent["color"],
    #             )
    #             self.plot.text(
    #                 position[1] + 1.2,
    #                 position[0] + 1.2,
    #                 f'{agent["role"]}-{agent_id} ({position[1]}, {position[0]})',
    #                 fontsize=8,
    #             )
    #             sensor_area = self.plot.Circle(
    #                 (position[1], position[0]),
    #                 agent["sensor"]["range"],
    #                 color=agent["color"],
    #                 fill=False,
    #             )
    #             self.plot.gca().add_artist(sensor_area)

    # def _add_targets_to_plot(self):
    #     for target_id, position in enumerate(self.targets["positions"]):
    #         self.plot.scatter(
    #             position[1],
    #             position[0],
    #             marker=self.targets["symbol"],
    #             color=self.targets["color"],
    #         )
    #         self.plot.text(
    #             position[1] + 1.2,
    #             position[0] + 1.2,
    #             f"{target_id}-({position[1]}, {position[0]})",
    #             fontsize=8,
    #             color=self.targets["color"],
    #         )

    # def _add_base_to_plot(self):
    #     self.plot.scatter(
    #         self.base["position"][1],
    #         self.base["position"][0],
    #         marker=self.base["symbol"],
    #         color=self.base["color"],
    #     )
    #     self.plot.text(
    #         self.base["position"][1] + 1.2,
    #         self.base["position"][0] + 1.2,
    #         f'Base ({self.base["position"][1]}, {self.base["position"][0]})',
    #         fontsize=8,
    #         color=self.base["color"],
    #     )


@hydra.main(config_path="./", config_name="setup", version_base="2.1.0")
def main(cfg: DictConfig):
    agents = cfg.agents
    targets = cfg.targets
    base = cfg.base
    metadata = cfg.metadata

    generator = DatasetGenerator(
        agents=agents, targets=targets, base=base, metadata=metadata
    )
    generator.sample()
    # generator.visualize()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
