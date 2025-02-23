from typing import Optional


def render_agent_or_instance(
    position: list[int],
    shape: str,
    agent_size: int = 20,
    observation_area_size: int = 50,
    color: str = "black",
    scale_factor: Optional[int] = None,
    has_observation: bool = True,
    show_text: bool = True,
    rendering_parts: list[str] = None,
):

    if scale_factor is None:
        scale_factor = 1

    if rendering_parts is None:
        rendering_parts = []

    transform_style = "transform: translate(-50%, -50%);"

    agent = f"""
        position: absolute;
        top: {position[1] * scale_factor}px;
        left: {position[0] * scale_factor}px;
        font-size: {agent_size}px;
        {transform_style};
        color: {color};
    """

    rendering_parts.append(f'<div style="{agent}">{shape}</div>')

    if has_observation:
        observation_area = f"""
            position: absolute;
            top: {position[1] * scale_factor}px;
            left: {position[0] * scale_factor}px;
            width: {observation_area_size * scale_factor}px;
            height: {observation_area_size * scale_factor}px;
            border: 2px solid black;
            border-radius: 50%;
            border-color: {color};
            {transform_style}
        """

        rendering_parts.append(f'<div style="{observation_area}"></div>')

    if show_text:
        text_label = f"""
                position: absolute;
                top: {position[1] * scale_factor - 15}px;
                left: {position[0] * scale_factor + 25}px;
                font-size: 12px;
                {transform_style}
            """

        rendering_parts.append(
            f'<div style="{text_label}">({position[0]}, {position[1]})</div>'
        )

    return "".join(rendering_parts)
