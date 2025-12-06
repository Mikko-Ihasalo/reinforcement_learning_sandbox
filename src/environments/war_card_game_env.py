from typing import Optional
import numpy as np
import gymnasium as gym


def default_reward_function(player_card: int, enemy_card: int) -> float:
    """Default reward function for the War card game."""
    if player_card > enemy_card:
        return 1.0  # Win
    elif player_card < enemy_card:
        return -1.0  # Lose
    else:
        return 0.0  # Tie


class WarCardGameEnv(gym.Env):
    def __init__(self, reward_at_end: bool = False):
        # keep existing state
        self.player1_cards = np.zeros(13, dtype=np.int8)
        self.player2_cards = np.zeros(13, dtype=np.int8)
        self.round_number = 0
        self.reward_function = default_reward_function

        # new: whether to only provide final reward
        self.reward_at_end = reward_at_end

        # new: per-episode scores and winner tracking
        self.player1_score = 0
        self.player2_score = 0
        self.last_winner: Optional[int] = None  # 1=player1, 2=player2, 0=tie
        self.winner_history: list[int] = []

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

        # reset scores and last winner
        self.player1_score = 0
        self.player2_score = 0
        self.last_winner = None

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

        # Determine round reward (always computed so scores can be accumulated)
        round_reward = self.reward_function(action, enemy_action)

        # Update per-episode scores based on card comparison (for final winner)
        if action > enemy_action:
            self.player1_score += 1
        elif action < enemy_action:
            self.player2_score += 1
        # ties leave scores unchanged

        # Decide what reward to return depending on reward_at_end
        if self.reward_at_end:
            # intermediate steps: zero reward; final step: return episode outcome
            if terminated:
                if self.player1_score > self.player2_score:
                    reward = 1.0
                    self.last_winner = 1
                elif self.player1_score < self.player2_score:
                    reward = -1.0
                    self.last_winner = 2
                else:
                    reward = 0.0
                    self.last_winner = 0
                self.winner_history.append(self.last_winner)
            else:
                reward = 0.0
        else:
            # per-round reward behavior (unchanged)
            reward = round_reward
            if terminated:
                # still set last_winner for consistency
                if self.player1_score > self.player2_score:
                    self.last_winner = 1
                elif self.player1_score < self.player2_score:
                    self.last_winner = 2
                else:
                    self.last_winner = 0
                self.winner_history.append(self.last_winner)

        return self._get_obs(), reward, terminated, False, self._get_info()


gym.register(
    id="WarCardGame-v0",
    entry_point="src.environments.card_game_env:WarCardGameEnv",
    max_episode_steps=13,
)
