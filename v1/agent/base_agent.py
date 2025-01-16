import numpy as np


class BaseDroneAgent:
    def __init__(
        self,
        area_size: tuple[int, int],
        object_map: dict[str, int],
        observation_area: tuple[int, int] = (9, 9),
    ):
        self.object_map = object_map
        self.area_size = area_size
        self.observation_area = observation_area
        self.targets_found = 0  # how many targets are found
        self.visited_map = np.full(
            (self.area_size[0], self.area_size[1]),
            self.object_map.get("NOT_VISITED_AREA"),
        )

    def update_visited_map(
        self, observation: np.ndarray, binary_observation_mask: np.ndarray
    ) -> None:
        """
        Updates the visited area with the observation in the regions specified by the binary observation mask.

        Parameters:
            visited_area (np.ndarray): The map of visited areas to be updated.
            observation (np.ndarray): The observed values to be placed on the visited area.
            binary_observation_mask (np.ndarray): A binary mask indicating where the observation applies.

        Returns:
            None: Updates are made directly to the visited_area.
        """
        # Find the bounding box of the mask
        rows, cols = np.where(binary_observation_mask == 1)
        if len(rows) == 0 or len(cols) == 0:
            return  # Mask is empty, nothing to update

        min_row, max_row = rows.min(), rows.max() + 1
        min_col, max_col = cols.min(), cols.max() + 1

        # Extract the target region in the mask
        mask_height = max_row - min_row
        mask_width = max_col - min_col

        # Ensure the observation aligns with the mask's region
        if observation.shape != (mask_height, mask_width):
            raise ValueError(
                "Observation shape does not match the size of the binary observation mask region."
            )

        # Update the visited area with the observation values
        self.visited_map[min_row:max_row, min_col:max_col] = observation
