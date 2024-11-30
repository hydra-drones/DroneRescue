import numpy as np
from PIL import Image, ImageColor


class Environment:
    def __init__(
        self,
        obstacles_map: dict[int, tuple[int, int]],
        color_map: dict[str, str],
        object_map: dict[str, str],
        area_size: tuple[int, int] = (100, 100),
        num_targets: int = 10,
        num_obstacles: int = 50,
    ):
        self.area_size = area_size  # one side side
        self.num_targets = num_targets  # number of targets
        self.num_obstacles = num_obstacles  # number of obstacles
        self.obstacles_map = obstacles_map
        self.color_map = color_map
        self.object_map = object_map
        self.done = False

        self.area = np.zeros((self.area_size[0], self.area_size[1]))
        self._generate_area()

    def _generate_area(self):
        # TODO : set attribute if None
        obstacles_x, obstacles_y = self.get_random_x_y(
            self.area_size[0], self.area_size[1], self.num_obstacles
        )
        target_point_x, target_point_y = self.get_random_x_y(
            self.area_size[0], self.area_size[1], self.num_targets
        )

        # ========== Add obstacles ==========
        for x, y in zip(obstacles_x, obstacles_y):
            sampled_obstacle = self.obstacles_map[
                np.random.choice(list(self.obstacles_map.keys()))
            ]
            self._insert_kernel(
                sampled_obstacle, (x, y), self.object_map.get("OBSTACLE")
            )

        # ========== Add targets ==========
        for x, y in zip(target_point_x, target_point_y):
            self._insert_kernel((3, 3), (x, y), self.object_map.get("TARGET_POINT"))

    def step(
        self,
        action: int,
        agent_position: tuple[int, int],
        observation_area: tuple[int, int],
        speed: int,
    ):
        if action == 0:  # up
            new_position = agent_position + np.array([speed * -1, 0])
        elif action == 1:  # down
            new_position = agent_position + np.array([speed * 1, 0])
        elif action == 2:  # left
            new_position = agent_position + np.array([0, speed * -1])
        elif action == 3:  # right
            new_position = agent_position + np.array([0, speed * 1])
        elif action == 4:  # stop
            new_position = agent_position + np.array([0, 0])
        else:
            # Just stay in place if action is unknown
            new_position = agent_position + np.array([0, 0])

        new_position = tuple(
            np.clip(new_position, 0, (self.area_size[0] - 1, self.area_size[1] - 1))
        )
        observation, observation_binary_mask, metadata = self._get_observation(
            new_position, observation_area
        )

        # Check if the agent doesn't hit the obstacle
        if self.area[new_position] == self.object_map.get("OBSTACLE"):  # obstacle
            self.done = True

        # End the episode if agent sees the target
        if np.any(observation == self.object_map.get("TARGET_POINT")):  # target point
            self.done = True

        return new_position, self.done, observation, observation_binary_mask, metadata

    def _get_observation(
        self, agent_position: tuple[int, int], observation_area: tuple[int, int]
    ) -> np.ndarray:
        start_row, end_row, start_col, end_col = self._calculate_edges_of_kernel(
            self.area, observation_area, agent_position
        )

        # Calculate agent position inside local observation area
        metadata = {
            "agent_position_in_local_observation": (
                agent_position[0] - start_row,  # x
                agent_position[1] - start_col,  # y
            )
        }

        observation = self.area[start_row:end_row, start_col:end_col]

        binary_mask = np.zeros(self.area_size)
        binary_mask[start_row:end_row, start_col:end_col] = 1
        return observation, binary_mask, metadata

    def get_env_size(self) -> int:
        return self.area_size

    def _insert_kernel(
        self, kernel_shape: tuple[int, int], point: tuple[int, int], fill_with: int
    ):
        start_row, end_row, start_col, end_col = self._calculate_edges_of_kernel(
            self.area, kernel_shape, point
        )
        self.area[start_row:end_row, start_col:end_col] = fill_with

    @staticmethod
    def get_random_x_y(
        x_max_size, y_max_size: int, num_of_samples: int
    ) -> list[int] | int:
        x = np.random.randint(0, y_max_size, num_of_samples)
        y = np.random.randint(0, x_max_size, num_of_samples)
        if num_of_samples == 1:
            x, y = x[0], y[0]
        return x, y

    @staticmethod
    def _calculate_edges_of_kernel(
        area: np.ndarray, kernel_shape: tuple[int, int], point: tuple[int, int]
    ):
        kernel_center = (kernel_shape[0] // 2, kernel_shape[1] // 2)
        start_row = max(point[0] - kernel_center[0], 0)
        end_row = min(point[0] + kernel_center[0] + 1, area.shape[0])
        start_col = max(point[1] - kernel_center[1], 0)
        end_col = min(point[1] + kernel_center[1] + 1, area.shape[1])
        return start_row, end_row, start_col, end_col

    def render_env(self, save_path):
        """
        Saves a 2D numpy array as a JPEG image using a specified color map.

        Parameters:
        - env_array: 2D numpy array where each cell represents a specific color based on `color_map`.
        - color_map: Dictionary mapping integer values to colors (e.g., {0: 'white', 1: 'red'}).
        - save_path: Path where the JPEG image will be saved.

        """
        height, width = self.area.shape
        img = Image.new("RGB", (width, height))
        color_of_object: dict[int, str] = {
            object_representation: self.color_map[object]
            for object, object_representation in self.object_map.items()
        }
        for y in range(height):
            for x in range(width):
                value = self.area[y, x]
                color = color_of_object.get(
                    value, "black"
                )  # Use 'black' as a fallback if value not in color_map
                img.putpixel((x, y), ImageColor.getrgb(color))

        img.save(save_path, format="JPEG")

    def render_observation(self, observation: np.ndarray, save_path: str):
        height, width = observation.shape
        img = Image.new("RGB", (width, height))
        color_of_object: dict[int, str] = {
            object_representation: self.color_map[object]
            for object, object_representation in self.object_map.items()
        }
        for y in range(height):
            for x in range(width):
                value = observation[y, x]
                color = color_of_object.get(
                    value, "black"
                )  # Use 'black' as a fallback if value not in color_map
                img.putpixel((x, y), ImageColor.getrgb(color))

        img.save(save_path, format="JPEG")

    def render_visited_map(self, visited_map: np.ndarray, save_path: str):
        height, width = visited_map.shape
        img = Image.new("RGB", (width, height))
        color_of_object: dict[int, str] = {
            object_representation: self.color_map[object]
            for object, object_representation in self.object_map.items()
        }
        for y in range(height):
            for x in range(width):
                value = visited_map[y, x]
                color = color_of_object.get(
                    value, "black"
                )  # Use 'black' as a fallback if value not in color_map
                img.putpixel((x, y), ImageColor.getrgb(color))

        img.save(save_path, format="JPEG")
