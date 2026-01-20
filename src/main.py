import yaml
from environments.war_card_game_env import GymEnvironment
from models.index import DQN, PPO
from agents.black_jack_agent import Agent


def main():
    # Load configuration
    with open("configs/config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    # Initialize environment
    env = GymEnvironment(config["environment"])

    # Initialize model
    model = DQN(config["model"])  # Example with DQN, can be changed to PPO or others

    # Initialize agent
    agent = Agent(model, env, config["agent"])

    # Start training or testing
    if config["mode"] == "train":
        agent.train()
    else:
        agent.test()


if __name__ == "__main__":
    main()
