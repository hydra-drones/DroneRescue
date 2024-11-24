def generate_system_prompt(object_map: dict) -> str:
    """
    Generates the static system-level prompt.

    Args:
        object_map: A dictionary mapping object types to their integer representations.
    """
    return f"""
        You are controlling an autonomous drone in a grid-based area.
        The grid contains visited and unvisited cells. Your task is to decide the next action.
        The possible actions are:
        0 - Move up
        1 - Move down
        2 - Move left
        3 - Move right
        4 - Stay in place

        The environment is the grid-world where each cell has defined digit. Each object has
        its own number. Here is the object map: {object_map}

        Rules:
        1. Avoid obstacles. If you hit an obstacle, you will fail the mission.
        2. Plan your path efficient
        3. If you will find the target flag - you win
    """


def generate_drone_state_prompt(visited_map, observation, observation_area_size) -> str:
    """
    Generates the dynamic drone state prompt.

    Args:
        visited_map: A 2D list representing the visited cells in the grid.
        observation: A 2D list representing the current observation area.
        observation_area_size: An integer representing the size of the observation area.
    """
    return f"""
        Here is the current state of the drone:
        Visited map: {visited_map}
        Observation area size: {observation_area_size}
        Current observation: {observation}
    """
