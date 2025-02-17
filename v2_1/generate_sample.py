from v2_1.variables import *
import numpy as np
from matplotlib import pyplot as plt


class DatasetGenerator:
    number_of_agents = None
    agent_positions = None
    number_of_targets = None
    target_positions = None
    roles = None

    def __init__(self, **args):
        self.number_of_agents: tuple[int] = args.get("NUMBER_OF_AGENTS")
        self.agent_positions: tuple[tuple[int]] = args.get("AGENT_POSITIONS")
        self.number_of_targets: tuple[int] = args.get("NUMBER_OF_TARGETS")
        self.roles: list[str] = args.get("ROLES")
        self.size_of_mission_area: tuple[int] = args.get("SIZE_OF_MISSION_AREA")
        self.agent_symbols: dict[str, str] = args.get("AGENT_SYMBOLS")
        self.agent_colors: dict[str, str] = args.get("AGENT_COLORS")

    def sample(self):
        num_roles = len(self.roles)

        # Ensure at least one agent per role
        min_agents = num_roles
        max_agents = self.number_of_agents[1]

        # Sample total number of agents
        num_agents = np.random.randint(min_agents, max_agents + 1)

        # Assign roles (ensuring each role is represented)
        roles = []
        remaining_agents = num_agents - num_roles
        roles.extend(self.roles)  # Ensure each role appears at least once
        roles.extend(np.random.choice(self.roles, remaining_agents).tolist())

        # Sample agent positions within the specified range
        y_min, x_min = self.agent_positions[0]
        y_max, x_max = self.agent_positions[1]
        agent_positions = [
            (np.random.randint(y_min, y_max + 1), np.random.randint(x_min, x_max + 1))
            for _ in range(num_agents)
        ]

        # Sample number of targets
        num_targets = np.random.randint(
            self.number_of_targets[0], self.number_of_targets[1] + 1
        )

        target_positions = [
            (np.random.randint(y_min, y_max + 1), np.random.randint(x_min, x_max + 1))
            for _ in range(num_targets)
        ]

        self.number_of_agents = num_agents
        self.agent_positions = agent_positions
        self.number_of_targets = num_targets
        self.target_positions = target_positions
        self.roles = roles

    def _get_symbol_by_agent(self, role):
        return self.agent_symbols[role]

    def _get_agent_color(self, role):
        return self.agent_colors[role]

    def visualize(self):
        agents = {}

        plt.figure(figsize=(6, 6))
        plt.xlim(0, self.size_of_mission_area[1])
        plt.ylim(0, SIZE_OF_MISSION_AREA[0])
        plt.gca().invert_yaxis()  # Invert the Y-axis
        plt.gca().xaxis.set_ticks_position("top")
        plt.grid(True)
        plt.xlabel("X")
        plt.ylabel("Y")

        for agent_id, agent_data in enumerate(zip(self.roles, self.agent_positions)):
            role, position = agent_data
            agents[agent_id] = {
                "role": role,
                "position": position,
                "symbol": self._get_symbol_by_agent(role),
                "color": self._get_agent_color(role),
            }

        for agent_id, agent in agents.items():
            plt.scatter(
                agent["position"][1],
                agent["position"][0],
                marker=agent["symbol"],
                color=agent["color"],
            )
            plt.text(
                agent["position"][1] + 1.2,
                agent["position"][0] + 1.2,
                f'{agent["role"]}-{agent_id} ({agent["position"][1]}, {agent["position"][0]})',
                fontsize=8,
            )

        for target in self.target_positions:
            targets = {}
            for target_id, target_position in enumerate(self.target_positions):
                targets[target_id] = {
                    "position": target_position,
                    "symbol": self._get_symbol_by_agent("target"),
                    "color": self._get_agent_color("target"),
                }

        for target_id, target in targets.items():
            plt.scatter(
                target["position"][1],
                target["position"][0],
                marker=target["symbol"],
                color=target["color"],
            )
            plt.text(
                target["position"][1] + 1.2,
                target["position"][0] + 1.2,
                f'{target_id}-({target["position"][1]}, {target["position"][0]})',
                fontsize=8,
                color=target["color"],
            )

        plt.savefig("sample.png")
        plt.close()


generator = DatasetGenerator(
    NUMBER_OF_AGENTS=NUMBER_OF_AGENTS,
    AGENT_POSITIONS=AGENT_POSITIONS,
    NUMBER_OF_TARGETS=NUMBER_OF_TARGETS,
    ROLES=ROLES,
    SIZE_OF_MISSION_AREA=SIZE_OF_MISSION_AREA,
    AGENT_SYMBOLS=AGENT_SYMBOLS,
    AGENT_COLORS=AGENT_COLORS,
)

generator.sample()
generator.visualize()
