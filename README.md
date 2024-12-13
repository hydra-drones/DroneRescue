# DroneRescue
Autonomous drone system for rescue and defense applications.

## Tools and standards used

- Project setup: [poetry](https://python-poetry.org/)
  - pyproject.toml
- [pre-commit](https://pre-commit.com/)
  - .pre-commit-config.yaml
- Formatting and linting [Ruff](https://github.com/astral-sh/ruff)
  - You can adjust the rules in .ruff.toml file
- Static Code Analysis [Pylance/Pyright](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)
  - Pylance is powered by [Pyright](https://github.com/microsoft/pyright/tree/main)
  - The configuration is set to 'standard' mode in: pyproject.toml
 
## Requirements

- Python 3.10
- Poetry
- Linux

## Setup

- `poetry install`
- `poetry shell`
- `pre-commit install`

## How to Run
- TODO


# Research Stage #1

## Takeaways from experiments

### Before making the decision
1. Agent takes into account moving direction of the teammate drone.
2. Agent is able to provide `team play` strategy
```
The observation shows an obstacle (1) to the southeast, and the teammate is moving south. We should support by moving right to explore unvisited areas and maintain formation.
```

### Communication with teammate drone
1. Drone informs about his own strategy
2. Send a request to keep the communication and inform about all important information will be found
```
Moving east to explore unvisited areas. Maintain formation and report any targets or obstacles.
```
