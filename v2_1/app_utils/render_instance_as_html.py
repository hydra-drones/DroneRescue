def render_agent_or_instance(
    position: list[int],
    shape: str,
    agent_size: int = 20,
    observation_area_size: int = 50,
    color: str = "black",
    has_observation: bool = True,
    show_text: bool = True,
    rendering_parts: list[str] = None,
):

    transform_style = "transform: translate(-50%, -50%);"

    agent = f"""
        position: absolute;
        top: {position[1]}px;
        left: {position[0]}px;
        font-size: {agent_size}px;
        {transform_style};
        color: {color};
    """

    if rendering_parts is None:
        rendering_parts = []

    rendering_parts.append(f'<div style="{agent}">{shape}</div>')

    if has_observation:
        observation_area = f"""
            position: absolute;
            top: {position[1]}px;
            left: {position[0]}px;
            width: {observation_area_size}px;
            height: {observation_area_size}px;
            border: 2px solid black;
            border-radius: 50%;
            border-color: {color};
            {transform_style}
        """

        rendering_parts.append(f'<div style="{observation_area}"></div>')

    if show_text:
        text_label = f"""
                position: absolute;
                top: {position[1] - 15}px;
                left: {position[0] + 25}px;
                font-size: 12px;
                {transform_style}
            """

        rendering_parts.append(
            f'<div style="{text_label}">({position[0]}, {position[1]})</div>'
        )

    return "".join(rendering_parts)
