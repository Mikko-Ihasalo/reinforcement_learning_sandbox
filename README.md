# Reinforcement Learning Sandbox Testing

This repository provides a sandbox environment for testing and experimenting with various reinforcement learning (RL) models and algorithms. It is designed to facilitate the development, evaluation, and comparison of different RL approaches in a structured manner.

## Project Structure

```
rl-sandbox-testing
├── src
│   ├── environments       # Contains environment definitions for RL models
│   ├── models             # Contains model definitions for RL algorithms
│   ├── agents             # Contains agent definitions that interact with environments and models
│   ├── utils              # Contains utility functions for various tasks
│   └── main.py            # Entry point for running experiments
├── tests                  # Contains unit tests for the components
├── configs                # Contains configuration settings for the project
├── scripts                # Contains scripts to run experiments
├── requirements.txt       # Lists Python dependencies required for the project
├── .gitignore             # Specifies files and directories to ignore by Git
└── README.md              # Documentation for the project
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd rl-sandbox-testing
pip install -r requirements.txt
```

## Usage

To run experiments, use the provided script:

```bash
python scripts/run_experiment.py
```

Make sure to configure the settings in `configs/config.yaml` according to your requirements.

## Components

- **Environments**: Implementations of various RL environments, encapsulated in classes such as `GymEnvironment`.
- **Models**: Definitions of RL algorithms, including implementations of models like `DQN` and `PPO`.
- **Agents**: Classes that manage the training and decision-making processes, such as the `Agent` class.
- **Utilities**: Helper functions for tasks like data preprocessing and logging.

## Testing

Unit tests are provided for each component in the `tests` directory. To run the tests, use:

```bash
pytest tests/
```

## TODO

Fix the AI slop

## License

This project is licensed under the MIT License. See the LICENSE file for details.