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


        To optimize the drone's movement for both time efficiency and safety, you can select
        a speed within the range of (1) to (9). Note that there is no "0" speed, as this would equate
        to a "stay in place" action. The speed you choose determines the exact number of cells the
        drone will move in the specified direction.

        For example:
        - If the speed is (6) and the action corresponds to moving downward, the drone will move (6) cells down**.

        Key considerations for choosing speed:
        - Obstacle Awareness: Be mindful of walls and other obstacles. The closer the drone is to a wall,
        the lower the speed should be to avoid collisions and failed actions.
        - High-Speed Scouting: Use higher speeds to quickly scout open areas or traverse long distances efficiently.
        - Precision Maneuvering: In tight spaces or near barriers, reduce the speed to maintain control and avoid errors.

        Choose your speed strategically to balance exploration and precision while minimizing risks.

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
