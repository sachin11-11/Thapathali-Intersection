# DQN Traffic Light Control System

Deep Q-Network (DQN) reinforcement learning system for optimizing traffic light control at the Thapathali intersection using SUMO simulation.

## Overview

This project implements a DQN-based traffic light controller that learns to minimize traffic congestion using the **Max-Pressure** reward function.

### Key Features

- **State Representation**: Per-lane vehicle counts + current phase (one-hot encoded)
- **Action Space**: 4 discrete traffic light phases
- **Reward Function**: Max-Pressure theory (negative total intersection pressure)
- **Phase Transitions**: 3-second yellow phase between switches
- **Action Duration**: 15-second green phases
- **Training**: 500 episodes with experience replay

## System Components

### 1. Configuration (`config.py`)
- Hyperparameters and training settings
- SUMO paths and traffic light IDs
- Phase definitions

### 2. SUMO Environment (`sumo_env.py`)
- Environment wrapper for SUMO TraCI interface
- Per-lane vehicle count state representation
- Max-pressure reward calculation
- Phase transition logic (yellow + green phases)

### 3. DQN Agent (`dqn_agent.py`)
- Neural network: 128 → 64 → 4 actions
- Experience replay buffer (10,000 samples)
- Epsilon-greedy exploration (ε: 1.0 → 0.01)
- Target network for stability

### 4. Training Script (`train_dqn.py`)
- Main training loop
- Reward tracking and visualization
- Model checkpointing

## Traffic Light Phases

- **Phase 0 (PHASE_1_STRAIGHTS)**: Maitighar ↔ Kupondole
- **Phase 1 (PHASE_2_MAITIGHAR_RIGHT)**: Maitighar → Tripureshwor + Kupondole
- **Phase 2 (PHASE_3_TRIPURESHWOR_STRAIGHT)**: Tripureshwor → Maternity + Left
- **Phase 3 (PHASE_4_KUPONDOLE_RIGHT)**: Kupondole → Maternity + Maitighar

## Installation

```bash
# Install Python dependencies
python3 -m pip install -r requirements.txt
```

## Usage

### Test the System
```bash
python3 test_system.py
```

### Train the DQN Agent
```bash
python3 train_dqn.py
```

This will:
- Train for 500 episodes
- Save checkpoints every 50 episodes to `trained_models/`
- Generate reward plot at `results/training_rewards.png`
- Save final model at `trained_models/dqn_traffic_light.pth`

### Visualize with SUMO GUI
Edit `config.py` and set:
```python
SUMO_GUI = True
```

## State Representation

The state is a vector of shape `(22,)`:
- **Lanes 0-5**: Incoming lane vehicle counts
- **Lanes 6-17**: Outgoing lane vehicle counts  
- **Lanes 18-21**: Current phase (one-hot: [1,0,0,0], [0,1,0,0], etc.)

## Reward Function

Based on **Max-Pressure Theory**:

```
R = -Σ (w(i) - 1/|L_out| * Σ w(j))
```

Where:
- `w(i)` = vehicle count on incoming lane `i`
- `w(j)` = vehicle count on outgoing lane `j`
- Encourages clearing crowded incoming lanes into available outgoing space

## Training Details

- **Episodes**: 500
- **Max Steps/Episode**: 3600 (1 hour simulation)
- **Learning Rate**: 0.001
- **Batch Size**: 64
- **Discount Factor (γ)**: 0.95
- **Epsilon Decay**: 0.995
- **Replay Buffer**: 10,000 experiences
- **Target Network Update**: Every 10 episodes

## Output Files

- `trained_models/dqn_traffic_light.pth` - Final trained model
- `trained_models/checkpoint_ep*.pth` - Intermediate checkpoints
- `results/training_rewards.png` - Reward progression plot

## Files Created

- **config.py** - Configuration and hyperparameters
- **sumo_env.py** - SUMO environment wrapper
- **dqn_agent.py** - DQN agent implementation
- **train_dqn.py** - Training script
- **test_system.py** - System verification script
- **requirements.txt** - Python dependencies
- **red_light_control.py** - Simple script to set all lights to red (testing)
