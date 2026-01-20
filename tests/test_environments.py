import unittest
from environments.war_card_game_env import GymEnvironment


class TestGymEnvironment(unittest.TestCase):

    def setUp(self):
        self.env = GymEnvironment()

    def test_environment_initialization(self):
        self.assertIsNotNone(self.env)

    def test_reset_environment(self):
        state = self.env.reset()
        self.assertIsNotNone(state)

    def test_step_function(self):
        state, reward, done, info = self.env.step(action=0)
        self.assertIsNotNone(state)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(done, bool)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
