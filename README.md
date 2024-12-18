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

## Experiment Nr. 1
In this experiment, we tested aagents in random generated environment. Agents was suited with following instructions:
1. Motion : agents have been informed about the available actions
2. Speed : agents have been prompted for choosing the speed for better performance
3. Object map : map with object description was provided
4. Communication : also agents were asked to communicate with the teammate agents to enhance the team play
5. Rules : into instruction the list of rules was included, which contained:
   - Avoidding the obstacles instruction
   - Planning movement path
   - The rule of success completion of the mission
7. Target : agents were informed about the main goal

### Takeaways from experiments

#### Communication
**Taking into account the teammate moving direction**
Agent takes into account movements of the teammate agent and plans his path respectively. For instance, if the teammate agent is moving right, then the agent makes the decision to move down to maximaze exploration efficiency. Below you can see the agent message for making the decision 
> The teammate is proceeding west at speed 6, focusing on unvisited areas. To complement their movement and maximize exploration efficiency, I'll move south at a moderate speed to explore unvisited areas while keeping a safe distance from potential obstacles.
> Proceeding westward at maximum speed to cover unvisited areas. Maintain your southward path. Let's synchronize our exploration to maximize coverage efficiency.

Quick decision changing
One of the main problem of current approach being changing the direction of movement on each step since we don't provide such kind of memory.

### Before making the decision
1. Agent takes into account moving direction of the teammate drone.

### Communication with teammate drone
1. Agent informs about his own strategy
2. Send a request to keep the communication and inform about all important information will be found
```
Moving east to explore unvisited areas. Maintain formation and report any targets or obstacles.
```

### Points of improvements
1. Movements are chaotic - agents don't take into account previous positions and explore already explored area
2. Agents don't understand where border of the area is
3. Agents crashe into a obstacles - ommiting obstalce mechanism doesn't work

