import unittest
from src.agents.index import Agent

class TestAgent(unittest.TestCase):

    def setUp(self):
        self.agent = Agent()

    def test_agent_initialization(self):
        self.assertIsNotNone(self.agent)

    def test_agent_action_selection(self):
        action = self.agent.select_action(state=[0, 0, 0])
        self.assertIn(action, self.agent.action_space)

    def test_agent_training(self):
        initial_state = [0, 0, 0]
        action = self.agent.select_action(initial_state)
        reward = 1.0
        next_state = [0, 1, 0]
        self.agent.train(initial_state, action, reward, next_state)
        # Add assertions to verify training logic if applicable

if __name__ == '__main__':
    unittest.main()