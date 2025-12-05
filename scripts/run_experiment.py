import yaml
import sys
from src.environments.index import GymEnvironment
from src.models.index import DQN, PPO
from src.agents.index import Agent

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    config = load_config(config_path)

    # Initialize environment
    env = GymEnvironment(config['environment'])

    # Initialize model
    if config['model']['type'] == 'DQN':
        model = DQN(config['model']['params'])
    elif config['model']['type'] == 'PPO':
        model = PPO(config['model']['params'])
    else:
        raise ValueError("Unsupported model type")

    # Initialize agent
    agent = Agent(model, env)

    # Run training or testing based on config
    if config['mode'] == 'train':
        agent.train(config['training'])
    elif config['mode'] == 'test':
        agent.test(config['testing'])
    else:
        raise ValueError("Unsupported mode")

if __name__ == "__main__":
    main()