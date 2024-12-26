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

### Video rollout

<img src="https://github.com/user-attachments/assets/33a40bcb-7e95-4e77-b1ad-f0880e853bc3" width="500"/>


## Experiment Nr. 2
In the following experiment, we added a long-term strategy variable. We instructed agents to generate a strategy, and then this strategy was propagated to the prompt. The agent decided to change or stick with the current strategy. We expected that the agent would be more sustainable and that this mechanism would prevent rapidly changing the direction of movement.

### Takeaways from experiments

**Overlapping of the visited area**  
Agents were also prompted to work efficiently and prevent overlapping, but it didn't work as we expected. One of the agents decided to align its movement strategy with the teammate agent. This is not a random situation; from the beginning, agents moved together in one direction—towards the right side, as observed in the following messages. Agents moved in one direction for 20 steps, which accounts for 1/5 of all steps in one experiment.

> To align with the teammate's movement and strategy, we should also move downward to explore the unvisited areas efficiently, complementing the teammate's path.  

> The right side has unvisited areas, and the teammate is moving right with speed 5. To maintain coverage and avoid redundancy, move right with a slightly lower speed to ensure safe maneuvering and area coverage.

On step 21, the first agent decided to change its direction and move down.  

> To avoid overlap with my teammate who is moving right, I will move downward to explore the unvisited areas below the current position. As there are no immediate obstacles in the downward direction, a moderate speed can be utilized for efficient exploration.

Also, we want to highlight that the agent tries to move efficiently and takes into account that the teammate agent is exploring the right side of the area.

> The right side is already being explored by the teammate, so we should move downward to explore new and unvisited areas while maintaining our coordinated coverage strategy.

### Finding target
In the current experiment, one of the agents found the target, and we want to analyze this event further. At some point, the agent informed that the target points were visible in the observation on the right side. The agent decided to move toward that target. Note: we didn't implement logic for the case when an agent finds the target because we wanted to observe what the agent would do in that situation.

> The drone has identified a cluster of target points directly to the right. Moving right will allow for target confirmation and potential mission progress.

The agent also informed its teammate agent about the target:

> Targets located to the right. Moving to investigate and confirm. Continue rightward exploration.

This agent then changed its long-term strategy to:

> Investigate the target points immediately to the right, then continue downward exploration. Collaborate with teammate for rightward coverage after confirming targets.

After investigation, the agent decided to continue the exploration to the right side and use high speed to explore efficiently:

> There is a large unvisited area to the right. To efficiently explore and confirm any targets, move right towards the unexplored sector while maintaining a safe distance from the obstacles to the left.  
> Proceeding to scout the right sector at high speed. Will update on any targets found.

### "Edge" problem
On evaluation, we observed that the first (left) agent, at the end of the rollout, started to move right. The messages from it suggested that during all the last steps, the agent was trying to investigate the right side. The agent doesn't know about edges and tries to investigate the entire area. This highlights a few problems: (1) lack of previous sequences and (2) lack of knowledge about the limits of the area. For now, we cannot determine the reason for this behavior. It might be easier to answer on this question if the agent didn't have the visited area (entire map), but it did.

### Observations
1. Implementing a long-term strategy helps reduce "chaotic" position changes and enhances the sustainability of the overall exploration.
2. Agents frequently adjusted their strategies to prevent overlapping, which is good, but they didn't take their own positions on the global map into account.
3. Agents tried to move right without considering the global map, neglecting the left side of the map—the reason for this is unknown.
4. Agents faced the "edge" problem. They don't know about edges and the bounds of the area.

### Points of improvement
1. Resolve the "edge" problem.
2. Change the decision-making mechanism to also focus on the global map, not just the current local observation.


### Video rollout

<img src="https://github.com/user-attachments/assets/227a4f44-08bf-4a3b-bb20-8cf6f9af637f" width="500"/>

# What should be improved - takeaways from Phase #1
1. Provide more accurate way to solve the problem of crashing into obstacles
2. Enable exchanging visited area with teammates to prevent overlapping
3. Don't allow agents to go outside the area of interest

# What was good
1. Long-term strategy. This is a really good feature which partly solves the problem of chaotic movements.
2. Summarizing and sending the important information to the teammate. Agents try to align their movements and move efficiently.

# Plans for the next phase
1. Provide the method to describe the visited (global) to be interpreted easily by agent
  - As an idea: the map can be divided into a global cells
  - Also, remember about area limit and provide undestanding of the area of interest
3. Provide module to send visited (explored) area to the teammate agent - update their own explored area
4. Improve the module for safety movements and prevent crashing into the obstacles
  - As an idea: we can provide the all potential position where the agent can go
6. These experiments don't provide a lot of insight according to communication abilities of the agents. We guess, it would be interesting to provide clear tasks and agents' "roles" to check the 
