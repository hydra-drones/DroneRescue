import numpy as np
from matplotlib import pyplot as plt
import hydra
from omegaconf import DictConfig


class DatasetGenerator:
    def __init__(self, **args):
        self.agents = args.get("agents")
        self.targets = args.get("targets")
        self.metadata = args.get("metadata")

    def _generate_agent_position(self, min_yx, max_yx):
        y_min, x_min = min_yx
        y_max, x_max = max_yx
        return np.random.randint(y_min, y_max + 1), np.random.randint(x_min, x_max + 1)

    def _generate_positions_for_n_agents(self, n, min_yx, max_yx):
        return [self._generate_agent_position(min_yx, max_yx) for _ in range(n)]

    def sample(self):
        for agent in self.agents.values():
            min_num, max_num = agent["number"]["min"], agent["number"]["max"]
            number_of_agents = np.random.randint(min_num, max_num + 1)
            agent["positions"] = self._generate_positions_for_n_agents(
                number_of_agents,
                agent["position_range"]["min_yx"],
                agent["position_range"]["max_yx"],
            )

        min_num, max_num = self.targets["number"]["min"], self.targets["number"]["max"]
        number_of_targets = np.random.randint(min_num, max_num + 1)
        self.targets["positions"] = self._generate_positions_for_n_agents(
            number_of_targets,
            self.targets["position_range"]["min_yx"],
            self.targets["position_range"]["max_yx"],
        )

    def visualize(self):
        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.metadata["size_of_mission_area"][1])
        plt.ylim(0, self.metadata["size_of_mission_area"][1])
        plt.gca().invert_yaxis()
        plt.gca().xaxis.set_ticks_position("top")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")

        for agent in self.agents.values():
            for agent_id, position in enumerate(agent["positions"]):
                plt.scatter(
                    position[1],
                    position[0],
                    marker=agent["symbol"],
                    color=agent["color"],
                )
                plt.text(
                    position[1] + 1.2,
                    position[0] + 1.2,
                    f'{agent["role"]}-{agent_id} ({position[1]}, {position[0]})',
                    fontsize=8,
                )

        for target_id, position in enumerate(self.targets["positions"]):
            plt.scatter(
                position[1],
                position[0],
                marker=self.targets["symbol"],
                color=self.targets["color"],
            )
            plt.text(
                position[1] + 1.2,
                position[0] + 1.2,
                f"{target_id}-({position[1]}, {position[0]})",
                fontsize=8,
                color=self.targets["color"],
            )

        plt.savefig("sample.png")
        plt.close()


@hydra.main(config_path="./", config_name="setup", version_base="2.1.0")
def main(cfg: DictConfig):
    agents = cfg.agents
    targets = cfg.targets
    metadata = cfg.metadata

    generator = DatasetGenerator(agents=agents, targets=targets, metadata=metadata)
    generator.sample()
    generator.visualize()


if __name__ == "__main__":
    main()
