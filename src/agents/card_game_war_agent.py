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


def card_value_difference(player_card: int, enemy_card: int) -> float:
    """TODO"""
    return enemy_card - player_card


def card_value_difference_win_loss(player_card: int, enemy_card: int) -> float:
    """TODO"""
    if player_card > enemy_card:
        return 1.0 / (0.5 + abs(enemy_card - player_card))  # Win
    elif player_card < enemy_card:
        return -1.0 / (0.5 + abs(enemy_card - player_card))  # Lose
    else:
        return 0.0  # Tie


class CardGameWarAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        reward_function=None,
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
        initial_q_values = initialze_q_values()
        self.q_values = initial_q_values

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.env = env

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.reward_function = reward_function

        self.training_error = []

    def change_reward_function(self, reward_function):
        self.reward_function = reward_function
        self.env.reward_function = reward_function

    def check_legal_cards(self, obs) -> list[int]:
        """Returns a list of legal cards that can be played."""
        player_cards = obs["player1_cards"]
        legal_cards = [card for card in range(13) if card not in player_cards]
        return legal_cards

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        Always returns the actual card index (0..12).
        """
        legal_actions = self.check_legal_cards(obs)

        # explore
        if np.random.random() < self.epsilon:
            return int(np.random.choice(legal_actions))

        # exploit: pick the legal action with the highest Q-value and return the actual card index
        q_vals = get_q_table_values(
            obs["round_number"],
            obs["player1_cards"],
            obs["player2_cards"],
            self.q_values,
        )
        # restrict to legal actions then pick best
        best_idx = int(np.argmax(q_vals[legal_actions]))
        return int(legal_actions[best_idx])

    def update(
        self, obs: dict, action: int, reward: float, terminated: bool, next_obs: dict
    ):
        """Updates the Q-value of an action."""

        next_legal_actions = self.check_legal_cards(next_obs)

        # future q should consider only legal next actions (if any)
        if not terminated and len(next_legal_actions) > 0:
            future_q_value = np.max(
                get_q_table_values(
                    next_obs["round_number"],
                    next_obs["player1_cards"],
                    next_obs["player2_cards"],
                    self.q_values,
                )[next_legal_actions]
            )
        else:
            future_q_value = 0.0

        current_q = get_q_table_values(
            obs["round_number"],
            obs["player1_cards"],
            obs["player2_cards"],
            self.q_values,
        )[action]

        temporal_difference = reward + self.discount_factor * future_q_value - current_q

        # update the Q-value for the actual action index
        get_q_table_values(
            obs["round_number"],
            obs["player1_cards"],
            obs["player2_cards"],
            self.q_values,
        )[action] = (
            current_q + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
