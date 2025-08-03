# Post-Harvest Storage Management using Reinforcement Learning

## Overview

This project aims to handle post-harvest crop loss due to inadequate storage. IIt aims to mitigate the use of chemical preservatives to protect stored crops, which may harm health, reduce crop quality, and lower market value. 

Throughtout this project, a custom Reinforcement Learning environment simulating dynamic storage conditions—temperature, humidity, pest levels was developed and used to train intelligent agents to learn when and how to apply chemical-free interventions across storage zones. Agents operate in a 3-phase loop: **Analyze → Navigate → Treat**.  

This system not only trains agents to make real-time decisions but also offers visual explanations and adaptive strategies based on the storage environment.

#### Demo

[📹 Click to view demo video](./demo.mp4)
---

## Custom Environment Description

### Agent Behavior

- The agent represents a digital technician operating inside a storage facility
- **Responsibilities**:
  - Analyzes environmental data (Temperature, Humidity, pest levels)
  - Navigates to the appropriate treatment zone
  - Applies the correct chemical-free intervention (e.g., neem oil, ventilation, solar drying)


---

###  Action Space

- **Type**: Discrete (22 actions)
- **Breakdown**:
  - `17 Treatment Actions`: Organic, physical, or biological (e.g., diatomaceous earth, ash, plant extracts)
  - `4 Navigation Actions`: Move Up, Down, Left, Right
  - `1 Condition Reading Action`: Start of the Analyze phase
- **Dynamic Action Masking**: Prevents illegal actions based on phase

---

### State Representation

- **Multi-modal Support**:
  - **Vector (MLP)**: Numerical array of 13 normalized values (temp, RH, pest level, position, zone, etc.)
  - **Image (CNN)**: 64×64 RGB rendering of the storage grid and agent status
  - **Dict Mode**: Combined observations with action masks and visual-state vectors
- **Designed for curriculum learning**, enabling agents to generalize across difficulty levels

---

### Reward Function

The agent is guided by a multi-level reward function tailored to the 3-phase structure:

- **Analyze Phase**:
  - +2: Correctly reads conditions
  - -0.5: Takes unrelated actions

- **Navigate Phase**:
  - +0.3: Moves closer to correct zone
  - -0.1 to -0.3: Drifts from target
  - +3: Reaches goal zone

- **Treat Phase**:
  - +8: Applies correct intervention
  - -4: Applies wrong intervention
  - -2: Applies treatment in wrong zone

- **Time Penalty**:
  - -0.02 per step to encourage efficient behavior

- **Timeout Penalty**:
  - -2 if episode ends without completion

---

### Visualization & Simulation

- **Developed with `pygame`** for real-time rendering
- **Features**:
  - 5×5 grid with semantic coloring (Critical, Safe, Too Dry, etc.)
  - Pest level meter and agent display
  - Visual phase cues (Blue = Analyze, Yellow = Navigate, Green = Treat)
  - Target zones flash after reading conditions
---

## Trained Algorithms

Four RL methods were evaluated on this environment:

| Algorithm | Description | Performance Summary |
|-----------|-------------|---------------------|
| **DQN** | Off-policy Q-learning with experience replay | Learns steadily but has slower convergence |
| **A2C** | On-policy actor-critic with entropy regularization | Fast learner, but needed careful tuning |
| **PPO** | Clipped Proximal Policy Optimization with GAE | Best performance, stable and sample-efficient |
| **REINFORCE** | Vanilla Monte Carlo policy gradient | Unstable without tuning, learned task phases but not semantics |

---

## Hyperparameter Insights

**PPO** stood out due to:
- `n_steps=2048`: Enables full-episode rollouts
- `clip_range=0.2`: Prevents abrupt policy shifts
- `net_arch=[256,256]`: Balances capacity and generalization

---

## Current Limitations & Future Work

- **Zone misclassification**: Some agents (e.g., PPO, A2C) occasionally choose incorrect zones due to misinterpreted conditions
- **No real wether data input**: Currently simulates weather and conditions
- **Only single-agent**: Could be extended to collaborative storage agents
- **Fixed environment layout**: Could be randomized during training for stronger generalization

### Future Enhancements
- Reward shaping: include bonus for correct risk prioritization
- Real wether data inputs from APIs or sensors