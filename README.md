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

```mermaid
gitGraph
   commit id: "init"

   branch dataset_base
   checkout dataset_base
   commit id: "data a"
   commit id: "data b"
   commit id: "data c"

   branch dataset_experiment
   checkout dataset_experiment
   commit id: "data 1"
   commit id: "data 2"
   checkout dataset_base
   merge dataset_experiment

   branch model_base
   checkout model_base
   commit id: "model a"
   commit id: "model b"
   commit id: "model c"

   branch model_experiment
   checkout model_experiment
   commit id: "model 1"
   commit id: "model 2"
   checkout model_base
   merge model_experiment

   branch dev
   checkout dev
   commit id: "dev 1"
   commit id: "dev 2"
   checkout model_base
   merge dev

   branch main
   checkout main
   commit tag: "0.1.0"
   checkout dev
   commit id: "dev 3"
   checkout main
   merge dev
   commit tag: "0.2.0"
```

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



## What Should Be Improved - Takeaways from Phase #1
1. Provide a more accurate method to solve the problem of crashing into obstacles.
2. Enable the exchange of visited areas with teammates to prevent overlapping.
3. Ensure agents do not go outside the area of interest.

## What Was Good
1. **Long-term strategy:** This is an excellent feature that partly addresses the problem of chaotic movements.
2. **Summarizing and sharing important information with teammates:** Agents attempt to align their movements and operate efficiently.

## Plans for the Next Phase
1. Develop a method to describe the visited (global) areas in a way that can be easily interpreted by agents.
   - Idea: The map can be divided into global cells.
   - Also, ensure the area limits are understood and clearly define the area of interest.
2. Create a module to send the visited (explored) area to teammate agents, enabling them to update their own explored areas.
3. Improve the safety movement module to prevent crashes into obstacles.
   - Idea: Provide all potential positions where an agent can move.
4. Address the lack of insight into agents’ communication abilities observed in these experiments. It may be interesting to define clear tasks and assign agents specific "roles" to test these capabilities.

==================================================

# Research Stage #2

## Experiment Description

In the current experiment, we decided to shift our focus to improving the communication system rather than movement control. This means that obstacles were excluded, as the goal in phase 2 was to study how agents cooperate with each other. A new feature introduced in this experiment is “roles.” Each agent now has a specific role in the mission. We experimented with two roles: “Scout” and “Rescuer.” Each role has a unique mission goal.

The Scout agent is tasked with searching for the target in predefined zones of interest—specific areas in the environment where a target is likely to be located. The Rescuer agent waits until the Scout agent sends the target coordinates. Upon receiving the coordinates, the Rescuer agent moves to the specified location and, upon arrival, returns to the starting point. This scenario simulates a real-life situation involving a catastrophic event that causes widespread destruction. In such a scenario, robots must locate individuals as quickly as possible within potential zones (e.g., positions of residential buildings extracted from a town map) and inform Rescuer robots to proceed to the designated rescue locations.

Additionally, in this experiment, we enhanced the visualization of the experiments. The visualization now has a higher resolution. We tracked the trajectories of each agent, and the visualization includes the areas of interest as well as the observation zones of each drone.

## Summary

**What has been changed:**

- Obstacles have been removed.
- Each agent now uses a Multi-Agent architecture.
- Roles have been introduced.
- Zones of interest have been added.

**Code improvements:**

- All code has been rewritten to adhere to Object-Oriented Programming (OOP) principles.
- The code is now clearer and easier to read.

**Visualization improvements:**

- Resolution has been increased.
- The full trajectory of each agent is now visualized.
- Zones of interest and agents’ observation zones are visualized.
- Messages have been added to the visualization.

## Multi-Agent Architecture

We implemented a multi-agent architecture to empower each agent with greater control and explainability. It is important to note that this approach is more computationally intensive.

### **Components**

1. **Descriptor**

    The Descriptor gathers all available information about the agent, including its position, observations, current strategy, and details about available teammate agents. It then provides a summary, referred to as the *state description*.

    **Output format**: (state_description)

2. **Decision Maker**

    The Decision Maker analyzes the state description and determines the next agent to call or engage.

    **Output format**: (message, next agent)

3. **Communicator**

    The Communicator facilitates message preparation for interaction with other agents. It generates messages based on inputs from the Decision Maker and selects the appropriate recipient.

    **Output format**: (recipient, message)

4. **Strategist**

    The Strategist manages and updates the agent’s current strategy. This component ensures that the agent maintains a consistent strategy across multiple evaluations, preventing fluctuations in behavior and reducing noise in agent movements.

    **Output format**: (new_strategy)

5. **Action**

    The Action component enables the agent to perform actions based on instructions from the Decision Maker and messages from other agents.

    **Output format**: (action, speed)

<img src="https://github.com/user-attachments/assets/9d5b98e6-0f8a-47f1-b1a3-d42d5b81a521" width="300"/>


## Experiment Analysis

I conducted approximately 50 experiments, all of which exhibited the same issues and behavior. Therefore, I will describe two of the experiments in detail.

### Success:

1. Since agents received information about their teammates, they are aware of each other's positions. This allows them to plan actions and strategies to avoid overlapping. As shown below, both agents, Green and Lime, move in different directions. This behavior is also evident in their messages:

    > *— Scoute Agent :*
    >
    >
    > Continue exploring sector 1 to locate potential targets. Ensure to avoid areas already covered by teammate 001
    >

   <img src="https://github.com/user-attachments/assets/ad69c40c-617f-4b7c-a3d4-504cdd29d346" width="800"/>

2. When the Scout agent detects a target, it immediately informs the Rescuer of the coordinates. However, the current solution has a problem: the agent sometimes sends its own current coordinates instead of the target's coordinates.

    > *— Scoute_001* sends to *Rescuer_002* following message:
    >
    >
    > Message 'Proceed to sector 2 at global position (1, 17) to investigate and handle the cluster of target points. Report back once completed.' has been sent to the 002 agent. Choose another agent
    >
    > *— Response of the Rescuer_002:*
    >
    > Strategy has been changed to new one : 'Investigate sector 2 at global position (1, 17) as instructed by the scout agent, prioritizing this task over returning to the start position. This adjustment is necessary to effectively handle the cluster of target points identified by the scout agent, ensuring mission objectives are met efficiently
    >

    Additionally, Scout_001 sends a message to its teammate, Scout_003, to inform them about the target

    > *— Scoute_001*:
    >
    >
    > Detected a cluster of target points in sector 2 at global position (9, 17). Please proceed to investigate and handle the targets.' has been sent to the 003 agent. Choose another agent.
    >

    When Scout_003 detects the target, it sends a message to Rescuer_002 with the target information.

    > — Scoute_003:
    >
    >
    > Message 'Detected target points in sector 4 at global position (13, 9). Please proceed to these coordinates and retrieve the targets. Confirm your battery level and position after completion.' has been sent to the 002 agent
    >

3. Once the Rescuer agent receives the target information, it moves to the specified position.

    > *— Rescuer_002*
    >
    >
    > Proceed to move towards the target point cluster in sector 2 at global position (1, 17) as instructed by the scout agent. This aligns with the current strategy to investigate and handle the cluster of target points efficiently
    >

    <img src="https://github.com/user-attachments/assets/8f843833-7110-449a-a488-7259c05a7c83" width="800"/>

## Issues:

- Agents do not understand when a task is completed and continue acting.
- The Rescuer does not interact with allies, making the mission impossible to complete.
- If the Rescuer does not respond, the team becomes paralyzed and remains in place.
- Agents do not prioritize tasks, leaving many of them unfinished.
- It is difficult to adapt the LLM for tasks requiring spatial understanding of the environment.

## Rollout


https://github.com/user-attachments/assets/6ba01a44-f1c4-492f-91ee-33c783fee5b6
