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

**Quick decision changing**

One of the main problem of current approach being changing direction of movement on each step. Hypothesis of the reasono of this behavior is that we don't provide such kind of memory. It's not expected behavior since each agent received visited map, which contain visited sections, so we expected taking into account this map and planning of work according to this map.

**Crash into the obstacles**

Current experiment shows that agents don't take into account ditstance to the obstacle. In experiment we provided to the model observation map, which can be visualized as a grid world, where each cell is a some variable, for instance to show the agent position we described the agent as a digit "4". In summary agent sees the obstacles, but it doesn't calculate the distance. We expected something like: "the distance to the next obstacle is 3, I should use low speed to investigate the area". But instead of it, there are cases when agent just going to the same direction where obstacle is which leads to the crashing. 
In the following example, agent noticed obstacle on the right side, but makes the decision to go right with speed 3. It was a mistake, since exactly after this action, the agent crashed into the obstacle. Note: for experimenting purposes, we don't finish rollout.

Message below is a message before crashing into the obstacle

> The current drone is positioned near a column of obstacles. Moving right will allow the drone to explore unvisited areas and avoid obstacles, while maintaining distance from the wall of obstacles

Following message is the message after crashing. In this state, the agent is already 'inside' the obstacle

> The current observation shows that there is a wall of obstacles directly to the right. To avoid collision and to continue exploring the unvisited areas, we should move left where the path is clear

### Points of improvements
1. Movements are chaotic - agents don't take into account previous positions and explore already explored area
2. Agents don't understand where border of the area is
3. Agents crashe into a obstacles - ommiting obstalce mechanism doesn't work

