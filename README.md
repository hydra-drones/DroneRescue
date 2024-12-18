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
In this experiment, we tested agents in a randomly generated environment. Agents were suited with the following instructions:
1. Motion: agents were informed about the available actions.
2. Speed: agents were prompted to choose the speed for better performance.
3. Object map: a map with object descriptions was provided.
4. Communication: agents were also asked to communicate with their teammate agents to enhance team play.
5. Rules: the instructions included a list of rules, which contained:
   - Avoiding obstacles.
   - Planning movement paths.
   - The rule for successful completion of the mission.
6. Target: agents were informed about the main goal.

### Takeaways from experiments

#### Communication

**Taking into account the teammate's moving direction**
Agents take into account the movements of their teammate agents and plan their paths accordingly. For instance, if a teammate agent is moving right, then the agent decides to move down to maximize exploration efficiency. Below you can see the agent's message for making the decision:

> The teammate is proceeding west at speed 6, focusing on unvisited areas. To complement their movement and maximize exploration efficiency, I'll move south at a moderate speed to explore unvisited areas while keeping a safe distance from potential obstacles.

> Proceeding westward at maximum speed to cover unvisited areas. Maintain your southward path. Let's synchronize our exploration to maximize coverage efficiency.

**Quick decision changing**

One of the main problems with the current approach is the frequent change of movement direction at each step. A hypothesis for the reason behind this behavior is the lack of memory. This is not expected behavior since each agent received a visited map containing visited sections. We expected them to take this map into account and plan their work accordingly.

**Crashing into obstacles**

The current experiment shows that agents don't take into account the distance to obstacles. In the experiment, we provided the model with an observation map, which can be visualized as a grid world, where each cell is a variable. For instance, to show the agent's position, we described the agent as the digit "4." In summary, the agent sees the obstacles, but it doesn't calculate the distance. We expected something like: "The distance to the next obstacle is 3; I should use low speed to investigate the area." But instead, there are cases where the agent simply continues in the same direction where the obstacle is, leading to a crash. 

In the following example, the agent noticed an obstacle on the right side but decided to go right with speed 3. It was a mistake since, immediately after this action, the agent crashed into the obstacle. Note: for experimental purposes, we don't terminate the rollout.

The message below is the message before crashing into the obstacle:

> The current drone is positioned near a column of obstacles. Moving right will allow the drone to explore unvisited areas and avoid obstacles, while maintaining distance from the wall of obstacles.

The following message is the message after crashing. In this state, the agent is already 'inside' the obstacle:

> The current observation shows that there is a wall of obstacles directly to the right. To avoid collision and to continue exploring the unvisited areas, we should move left where the path is clear.

### Points of improvement
1. Movements are chaotic - agents don't take into account previous positions and explore already explored areas.
2. Agents don't understand where the border of the area is.
3. Agents crash into obstacles - the obstacle avoidance mechanism doesn't work.
