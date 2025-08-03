# Post-Harvest Storage Management using Reinforcement Learning

## Overview

This project aims to handle post-harvest crop loss due to inadequate storage. IIt aims to mitigate the use of chemical preservatives to protect stored crops, which may harm health, reduce crop quality, and lower market value. 

Throughtout this project, a custom Reinforcement Learning environment simulating dynamic storage conditions—temperature, humidity, pest levels was developed and used to train intelligent agents to learn when and how to apply chemical-free interventions across storage zones. Agents operate in a 3-phase loop: **Analyze → Navigate → Treat**.  

This system not only trains agents to make real-time decisions but also offers visual explanations and adaptive strategies based on the storage environment.

## Demo

Link to demo video: https://drive.google.com/file/d/1gddwCn-aW05aWHdsmZCRPc7tRKPo9XGk/view?usp=sharing

## Environment Description

### Agent

The agent represents a storage management specialist with capabilities to:
- Analyze environmental conditions (temperature, humidity, pest levels)
- Navigate through a 5×5 storage facility
- Apply appropriate natural treatments
- Perform multi-phase decision making with phase-appropriate actions

### Action Space

22 discrete actions categorized into:

1. **Storage Management Actions (17):**
   - Ventilation control
   - Moisture management
   - Natural pest treatments (neem, wood ash, etc.)
   - Emergency measures

2. **Navigation Actions (4):** Move up, down, left, right

3. **Analysis Action (1):** Read environmental conditions

**Action Masking:** Only phase-appropriate actions are valid:
- ANALYZE: Only reading conditions
- NAVIGATE: Only movement
- TREAT: Only zone-specific treatments

### State Space

Three observation modes:

1. **MLP Observation (Vector):** 13-dimensional normalized vector
   - Environmental conditions
   - Storage metadata
   - Spatial information
   - Episode progress
   - Phase encoding

2. **CNN Observation:** 64×64×3 RGB image
   - Color-coded zones
   - Agent position
   - Target highlighting

3. **Multi-modal Observation:** Dictionary combining vector, image, and action mask

### Reward Structure

Multi-component reward system with:

- **Phase-based rewards:**
  - ANALYZE: +2.0 for reading conditions
  - NAVIGATE: Progress rewards (+0.3 closer, +3.0 correct zone)
  - TREAT: +8.0 correct treatment, -4.0 incorrect

- **General penalties:**
  - Time penalty: -0.02/step
  - Invalid action: -1.0

- **Curriculum learning:** Adapts to three difficulty stages

## Implemented Methods

### Deep Q-Network (DQN)
- Architecture: [256, 256, 128] fully connected network
- Features: Experience replay (10,000 samples), target network updates
- Exploration: ε-greedy (1.0→0.05 over 30% training)
- Learning rate: 1e-3, γ=0.99

### Policy Gradient Methods

#### Proximal Policy Optimization (PPO)
- Actor-critic with [256, 256] hidden units
- Clip range: 0.2, GAE λ=0.95
- Rollout buffer: 2,048 steps, 10 optimization epochs
- Learning rate: 3e-4

#### Advantage Actor-Critic (A2C)
- Similar architecture to PPO
- n_steps=20, GAE λ=0.95
- Learning rate: 5e-4

#### REINFORCE
- Pure policy gradient (no critic)
- High entropy coefficient (0.3)
- Learning rate: 1e-4

## Hyperparameter Optimization

Key findings from hyperparameter tuning:

| Method | Optimal Parameters | Impact |
|--------|--------------------|--------|
| **DQN** | Batch size=64, ε-decay over 30% training | Stable but slow learning |
| **A2C** | n_steps=20, ent_coef=0.05 | Reduced oscillation issues |
| **PPO** | n_steps=2048, clip_range=0.2 | Most stable performance |
| **REINFORCE** | ent_coef=0.3, lr=1e-4 | Improved exploration |

## Results

**Performance Comparison:**

| Metric | PPO | A2C | DQN | REINFORCE |
|--------|-----|-----|-----|-----------|
| Cumulative Reward | High | Medium | Low | Variable |
| Training Stability | Best | Moderate | Good | Poor |
| Generalization | Excellent | Good | Fair | Poor |
| Semantic Understanding | Yes | Partial | Limited | No |

**Key Findings:**
- PPO achieved best overall performance with stable learning and semantic understanding
- A2C learned quickly but showed oscillation behavior
- DQN was stable but slow to converge
- REINFORCE struggled with high variance

## Conclusion

PPO emerged as the most effective method for this environment, demonstrating:
- Stable policy updates
- Effective learning of the three-phase task structure
- Good generalization to new scenarios
- Semantic understanding of zone conditions

Future improvements could include:
- Enhanced treatment selection logic
- Dynamic weather conditions
- More sophisticated reward shaping

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- Gymnasium
- NumPy
- Pygame (for visualization)

### Installation
```bash
git clone https://github.com/Chrisos10/Jean_Chrisostome_Dufitumukiza_rl_summative.git
cd Jean_Chrisostome_Dufitumukiza_rl_summative
pip install -r requirements.txt

## Current Limitations & Future Work

- **Zone misclassification**: Some agents (e.g., PPO, A2C) occasionally choose incorrect zones due to misinterpreted conditions
- **No real wether data input**: Currently simulates weather and conditions
- **Only single-agent**: Could be extended to collaborative storage agents
- **Fixed environment layout**: Could be randomized during training for stronger generalization

### Future Enhancements
- Reward shaping: include bonus for correct risk prioritization
- Real wether data inputs from APIs or sensors