import unittest
from drone_rescue_ai.experiments.environment import DroneEnv
import numpy as np


class TestSimpleDotGameEnv(unittest.TestCase):
    def setUp(self):
        self.env = DroneEnv()

    def test_initial_state(self):
        obs = self.env.reset()
        self.assertTrue((obs == np.array([1, 1])).all())
        self.assertFalse(self.env.done)

    def test_step_up(self):
        self.env.reset()
        obs, reward, done, _ = self.env.step(0)  # Move up
        expected_position = np.array([1, 0])
        self.assertTrue((obs == expected_position).all())
        self.assertAlmostEqual(reward, -0.1)
        self.assertFalse(done)

    def test_step_to_obstacle(self):
        self.env.reset()
        self.env.player_position = np.array([3, 2])  # Set position near an obstacle
        obs, reward, done, _ = self.env.step(1)  # Move down to hit obstacle
        self.assertTrue(done)
        self.assertAlmostEqual(reward, -1)

    def test_reach_flag(self):
        self.env.reset()
        self.env.player_position = np.array([8, 7])  # Set position near the flag
        obs, reward, done, _ = self.env.step(1)  # Move down to reach the flag
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 10)

    def test_go_out_from_the_range(self):
        self.env.reset()
        self.env.player_position = np.array([1, 1])
        _, reward, done, _ = self.env.step(0)
        print(self.env.player_position)
        self.env.render()


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
