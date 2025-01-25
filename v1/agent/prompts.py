def generate_system_prompt(object_map: dict, total_targets: int) -> str:
    """
    Generates the static system-level prompt.

    Args:
        object_map: A dictionary mapping object types to their integer representations.
    """
    return f"""
        # Initialization
        You are controlling an autonomous drone in a grid-based area.

        # Motion
        The grid contains visited and unvisited cells. Your task is to decide the next action.
        The possible actions are:
        0 - Move up
        1 - Move down
        2 - Move left
        3 - Move right
        4 - Stay in place

        # Speed
        To optimize the drone's movement for both time efficiency and safety, you can select
        a speed within the range of (1) to (9). Note that there is no "0" speed, as this would equate
        to a "stay in place" action. The speed you choose determines the exact number of cells the
        drone will move in the specified direction.

        For example:
        - If the speed is (6) and the action corresponds to moving downward, the drone will move (6) cells down.

        Key considerations for choosing speed:
        - Obstacle Awareness: Be mindful of walls and other obstacles. The closer the drone is to a wall,
        the lower the speed should be to avoid collisions and failed actions.
        - High-Speed Scouting: Use higher speeds to quickly scout open areas or traverse long distances efficiently.
        - Precision Maneuvering: In tight spaces or near barriers, reduce the speed to maintain control and avoid errors.

        Choose your speed strategically to balance exploration and precision while minimizing risks.

        # Environment
        The environment is the grid-world where each cell has defined digit. Each object has
        its own number. Here is the object map: {object_map}

        # Communication with team
        During the mission you can send the message to your teammate.
        Include all the critical information that
        would be useful for the coordination and operation of a
        drone swarm, such as position, status, environmental data,
        task progress, or specific commands (orders).
        Remember that you also can send to your teammate information
        like a visited map or your current strategy

        # Restrctions
        You should plan your path and strategy carefully, taking into account your battery charge level.
        With every step (regardless of your speed), your battery will discharge.
        Ensure your actions are efficient to complete the mission before your battery is depleted.

        # Long-term Strategy
        Plan a long-term exploration strategy by identifying specific sectors or areas of interest
        to prioritize, such as 'Check right side sector' or unexplored regions. Your strategy should focus on efficient
        coverage of the environment and ensure systematic exploration, avoiding redundant paths

        # Obstacles
        During the exploration you will meet the obstacles - you have to avoid them to prevent crash!
        If you crash, your mission will be failed

        # Rule
        Remember don't visit the area which has been already visited. You can save more energy to
        search new sectors!

        # Rule 2
        Also, remember that the area is limited to 100x100 cells, and when you
        are close to the border, it might make sense to change your direction.

        # Your target
        Your team's mission is to locate and identify {total_targets} targets.
        Once your team has successfully located all the targets, the mission is complete, and you win!
    """


def generate_drone_state_prompt(
    visited_map,
    observation,
    observation_area_size,
    message_from_drone,
    long_term_strategy,
    n_previous_actions,
    n_previous_speed,
) -> str:
    """
    Generates the dynamic drone state prompt.

    Args:
        visited_map: A 2D list representing the visited cells in the grid.
        observation: A 2D list representing the current observation area.
        observation_area_size: An integer representing the size of the observation area.
    """
    return f"""
        Here is the current state of the drone:
        Observation area size: {observation_area_size}
        Visited map: {visited_map}
        Your last actions : {n_previous_actions}
        Yout last speed: {n_previous_speed}
        Current observation: {observation}
        Message from your teammate drone: {message_from_drone}
        Your long-term strategy: {long_term_strategy}
    """
