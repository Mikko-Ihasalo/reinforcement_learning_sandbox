from typing import Optional
import numpy as np
import gymnasium as gym


class WarCardGameEnv(gym.Env):
    def __init__(self):
        self.player1_cards = np.zeros(13, dtype=np.int8)
        self.player2_cards = np.zeros(13, dtype=np.int8)
        self.round_number = 0

        self.observation_space = gym.spaces.Dict(
            {
                "player1_cards": gym.spaces.Box(
                    low=0, high=1, shape=(13,), dtype=np.int8
                ),  # The "real" player will always be the player1, enemy will be player2.  We play with cards from 0 to 12
                "player2_cards": gym.spaces.Box(
                    low=0, high=1, shape=(13,), dtype=np.int8
                ),
                "round_number": gym.spaces.Discrete(
                    14
                ),  # 13 round + 1 for the round before the game starts.
            }
        )
        self.action_space = gym.spaces.Discrete(13)  # Play a card from 0 to 12

    def _get_obs(self) -> dict:
        return {
            "player1_cards": np.where(self.player1_cards == 1)[0].tolist(),
            "player2_cards": np.where(self.player2_cards == 1)[0].tolist(),
            "round_number": self.round_number,
        }

    def _get_info(self):
        return {
            "player1_played_cards": np.where(self.player1_cards == 1)[0].tolist(),
            "player2_played_cards": np.where(self.player2_cards == 1)[0].tolist(),
            "round_number": self.round_number,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.player1_cards = np.zeros(13, dtype=np.int8)
        self.player2_cards = np.zeros(13, dtype=np.int8)
        self.round_number = 0

        observation = self._get_obs()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Takes an action (play a card) based on the number given and returns the new observation, reward, done, and info."""

        self.round_number += 1
        terminated = self.round_number >= 13

        assert self.action_space.contains(
            action
        ), f"Invalid action {action}, action must be between 0 and 12"

        available_enemy_cards = np.where(self.player2_cards == 0)[0]
        enemy_action = np.random.choice(available_enemy_cards)
        self.player2_cards[enemy_action] = 1

        self.player1_cards[action] = 1

        # Enemy plays a random card that hasn't been played yet
        # TODO: Create a smarter enemy later. For now, it's just random.

        # Determine reward
        if action > enemy_action:
            reward = 1  # Win
        elif action < enemy_action:
            reward = -1  # Lose
        else:
            reward = 0  # Tie

        return self._get_obs(), reward, terminated, False, self._get_info()


gym.register(
    id="WarCardGame-v0",
    entry_point="src.environments.card_game_env:WarCardGameEnv",
    max_episode_steps=13,
)
