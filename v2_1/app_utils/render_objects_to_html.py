from typing import Dict
from v2_1.app_utils.render_instance_as_html import render_agent_or_instance
from v2_1.generate_sample import AgentData, TargetData

# TODO: add typing for agent_metadata
def render_agent(
    agents: Dict[int, AgentData], agent_metadata: dict, scale_factor: int = 1
) -> str:
    rendering_parts = []
    for agent_id, agent in agents.items():
        rendering_parts.append(
            render_agent_or_instance(
                agent.position,
                f'&#x{agent_metadata[agent.role]["shape"]};',
                agent_metadata[agent.role]["size"],
                agent_metadata[agent.role]["sensor"]["range"],
                agent_metadata[agent.role]["color"],
                agent_id,
                scale_factor,
            )
        )
    return " ".join(rendering_parts)


def render_target(
    targets: Dict[int, TargetData], target_metadata: dict, scale_factor: int = 1
) -> list[str]:
    rendering_parts = []
    for target_id, target in targets.items():
        rendering_parts.append(
            render_agent_or_instance(
                target.position,
                f'&#x{target_metadata["shape"]};',
                target_metadata["size"],
                0,
                target_metadata["color"],
                target_id,
                scale_factor,
                has_observation=False,
            )
        )

    return "".join(rendering_parts)


def render_base(
    bases: Dict[int, TargetData], base_metadata: dict, scale_factor: int = 1
) -> str:
    rendering_parts = []
    for base_id, base in bases.items():
        rendering_parts.append(
            render_agent_or_instance(
                base.position,
                f'&#x{base_metadata["shape"]};',
                base_metadata["size"],
                0,
                base_metadata["color"],
                base_id,
                scale_factor,
                has_observation=False,
                show_text=False,
            )
        )
    return "".join(rendering_parts)
