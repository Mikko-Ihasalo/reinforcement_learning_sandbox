import numpy as np
import math as math
import itertools


def initialze_q_values() -> dict[int, np.ndarray]:
    q_table = {}
    for i in range(0, 13):
        dim = math.comb(13, i)
        q_table[i] = np.zeros(
            (dim, dim, 13)
        )  # This could be optimized to only legal actions in the 3rd dimension, but for debugging we keep the order of actions the same
    return q_table


def get_cards_played_index(cards: list[int]) -> int:
    """
    Returns the index of the given cards combination in a sorted list of all possible combinations.

    Args:
        cards: List of card indices (1-13)

    Returns:
        Index of this combination in the sorted list of all combinations of the same size
    """
    n = len(cards)
    # Generate all combinations of size n from cards 1-13
    all_combinations = list(itertools.combinations(range(13), n))

    # Sort combinations (should already be sorted by itertools.combinations)
    all_combinations.sort()

    # Convert cards to tuple and find its index
    cards_tuple = tuple(sorted(cards))
    return all_combinations.index(cards_tuple)


def get_q_table_values(
    round_number: int,
    player1_cards: list[int],
    player2_cards: list[int],
    q_table: dict[int, np.ndarray],
) -> np.ndarray:
    player1_index = get_cards_played_index(player1_cards)
    player2_index = get_cards_played_index(player2_cards)
    return q_table[round_number][player1_index, player2_index, :]


class CardGameWarAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.q_values = initialze_q_values()

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.env = env

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def check_legal_cards(self, obs) -> list[int]:
        """Returns a list of legal cards that can be played."""
        player_cards = obs["player1_cards"]
        legal_cards = [card for card in range(13) if card not in player_cards]
        return legal_cards

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment

        legal_actions = self.check_legal_cards(obs)

        if np.random.random() < self.epsilon:
            return np.random.choice(legal_actions)

        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(
                np.argmax(
                    get_q_table_values(
                        obs["round_number"],
                        obs["player1_cards"],
                        obs["player2_cards"],
                        self.q_values,
                    )[legal_actions]
                )
            )

    def update(
        self, obs: dict, action: int, reward: float, terminated: bool, next_obs: dict
    ):
        """Updates the Q-value of an action."""
        print("Updating Q-values")
        print(
            f"Obs: {obs}, Action: {action}, Reward: {reward}, Terminated: {terminated}, Next_obs: {next_obs}"
        )
        legal_actions = self.check_legal_cards(next_obs)

        future_q_value = (not terminated) * np.max(
            get_q_table_values(
                next_obs["round_number"],
                next_obs["player1_cards"],
                next_obs["player2_cards"],
                self.q_values,
            )
        )
        temporal_difference = (
            reward
            + self.discount_factor * future_q_value
            - get_q_table_values(
                obs["round_number"],
                obs["player1_cards"],
                obs["player2_cards"],
                self.q_values,
            )[legal_actions][action]
        )

        get_q_table_values(
            obs["round_number"],
            obs["player1_cards"],
            obs["player2_cards"],
            self.q_values,
        )[legal_actions][action] = (
            get_q_table_values(
                obs["round_number"],
                obs["player1_cards"],
                obs["player2_cards"],
                self.q_values,
            )[legal_actions][action]
            + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
