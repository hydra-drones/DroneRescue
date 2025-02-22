from typing import Dict
from v2_1.app_utils.render_instance_as_html import render_agent_or_instance
from v2_1.generate_sample import AgentData, TargetData

# TODO: add typing for agent_metadata
def render_agent(agents: Dict[int, AgentData], agent_metadata: dict) -> str:
    rendering_parts = []
    for agent in agents.values():
        rendering_parts.append(
            render_agent_or_instance(
                agent.position,
                f'&#x{agent_metadata[agent.role]["shape"]};',
                agent_metadata[agent.role]["size"],
                agent_metadata[agent.role]["sensor"]["range"],
                agent_metadata[agent.role]["color"],
            )
        )
    return " ".join(rendering_parts)


def render_target(targets: Dict[int, TargetData], target_metadata: dict) -> list[str]:
    rendering_parts = []
    for target in targets.values():
        rendering_parts.append(
            render_agent_or_instance(
                target.position,
                f'&#x{target_metadata["shape"]};',
                target_metadata["size"],
                0,
                target_metadata["color"],
                has_observation=False,
            )
        )

    return "".join(rendering_parts)


def render_base(bases: Dict[int, TargetData], base_metadata: dict) -> str:
    rendering_parts = []
    for base in bases.values():
        rendering_parts.append(
            render_agent_or_instance(
                base.position,
                f'&#x{base_metadata["shape"]};',
                base_metadata["size"],
                0,
                base_metadata["color"],
                has_observation=False,
                show_text=False,
            )
        )
    return "".join(rendering_parts)
