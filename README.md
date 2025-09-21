# Proximal Policy Optimization (PPO) Algorithm

A comprehensive implementation of the Proximal Policy Optimization algorithm for reinforcement learning, focusing on efficient policy optimization and loss minimization.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Algorithm Details](#algorithm-details)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

## Overview

Proximal Policy Optimization (PPO) is a policy gradient method for reinforcement learning developed by OpenAI. This implementation provides a robust and efficient solution for training agents in various environments while maintaining policy stability through clipped objective functions.

### Key Advantages of PPO

- **Sample Efficiency**: Better sample efficiency compared to traditional policy gradient methods
- **Stability**: Prevents large policy updates that could destabilize training
- **Simplicity**: Easier to implement and tune than other advanced methods like TRPO
- **Versatility**: Works well across a wide range of continuous and discrete action spaces

## Features

- ✅ Standard PPO with clipped surrogate objective
- ✅ Support for both continuous and discrete action spaces
- ✅ Generalized Advantage Estimation (GAE)
- ✅ Value function approximation with bootstrapping
- ✅ Adaptive KL penalty (optional)
- ✅ Multi-environment parallel training
- ✅ Comprehensive logging and monitoring
- ✅ GPU acceleration support
- ✅ Configurable network architectures
- ✅ Built-in environment wrappers

## Installation

### Prerequisites

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Gym/Gymnasium
- (Optional) CUDA for GPU acceleration

### Install from Source

```bash
git clone https://github.com/UdulaRana/Proximal-Policy-Optimization-Algorithm.git
cd Proximal-Policy-Optimization-Algorithm
pip install -r requirements.txt
```

### Using pip (when available)

```bash
pip install ppo-algorithm
```

## Quick Start

Here's a simple example to get you started:

```python
from ppo import PPOAgent, Environment

# Create environment
env = Environment('CartPole-v1')

# Initialize PPO agent
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
    lr_actor=3e-4,
    lr_critic=1e-3,
    gamma=0.99,
    eps_clip=0.2
)

# Train the agent
agent.train(env, episodes=1000)

# Test the trained agent
agent.test(env, episodes=10)
```

## Algorithm Details

### Mathematical Formulation

PPO optimizes the following clipped surrogate objective:

```
L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

Where:
- `r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)` is the probability ratio
- `Â_t` is the advantage estimate at time t
- `ε` is the clipping parameter (typically 0.1 or 0.2)

### Key Components

1. **Actor Network**: Policy function π(a|s) that outputs action probabilities
2. **Critic Network**: Value function V(s) that estimates state values
3. **Advantage Estimation**: Uses GAE for bias-variance tradeoff
4. **Clipped Objective**: Prevents large policy updates

### Training Process

1. **Collect Experience**: Gather trajectories using current policy
2. **Compute Advantages**: Calculate advantage estimates using GAE
3. **Update Policy**: Optimize clipped surrogate objective
4. **Update Value Function**: Minimize value prediction error
5. **Repeat**: Continue until convergence

## Usage Examples

### Custom Environment

```python
import gym
from ppo import PPOAgent

# Create custom environment
env = gym.make('LunarLander-v2')

# Configure agent
config = {
    'state_dim': env.observation_space.shape[0],
    'action_dim': env.action_space.n,
    'lr_actor': 3e-4,
    'lr_critic': 1e-3,
    'gamma': 0.99,
    'eps_clip': 0.2,
    'k_epochs': 4,
    'batch_size': 64
}

agent = PPOAgent(**config)
agent.train(env, episodes=2000)
```

### Continuous Action Space

```python
import gym
from ppo import ContinuousPPOAgent

env = gym.make('BipedalWalker-v3')

agent = ContinuousPPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_std=0.5,
    lr_actor=3e-4,
    lr_critic=1e-3
)

agent.train(env, episodes=5000)
```

### Multi-Environment Training

```python
from ppo import MultiEnvPPOAgent
import gym

# Create multiple environments
envs = [gym.make('CartPole-v1') for _ in range(8)]

agent = MultiEnvPPOAgent(
    state_dim=envs[0].observation_space.shape[0],
    action_dim=envs[0].action_space.n,
    n_envs=len(envs)
)

agent.train_parallel(envs, total_timesteps=100000)
```

## API Reference

### PPOAgent Class

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_dim` | int | Required | Dimension of state space |
| `action_dim` | int | Required | Dimension of action space |
| `lr_actor` | float | 3e-4 | Learning rate for actor network |
| `lr_critic` | float | 1e-3 | Learning rate for critic network |
| `gamma` | float | 0.99 | Discount factor |
| `eps_clip` | float | 0.2 | Clipping parameter |
| `k_epochs` | int | 4 | Number of optimization epochs |
| `batch_size` | int | 64 | Mini-batch size |

#### Methods

- `train(env, episodes)`: Train the agent in the given environment
- `test(env, episodes)`: Test the trained agent
- `save(path)`: Save the model to specified path
- `load(path)`: Load model from specified path
- `get_action(state)`: Get action for given state
- `update()`: Perform policy and value function updates

### Configuration Options

```python
config = {
    # Network Architecture
    'hidden_dim': 64,
    'n_layers': 2,
    'activation': 'tanh',
    
    # Training Parameters
    'max_episodes': 1000,
    'max_timesteps': 200,
    'update_timestep': 2000,
    
    # PPO Parameters
    'eps_clip': 0.2,
    'k_epochs': 4,
    'gamma': 0.99,
    'lambda_gae': 0.95,
    
    # Learning Rates
    'lr_actor': 3e-4,
    'lr_critic': 1e-3,
    'lr_decay': 0.99,
    
    # Logging
    'log_interval': 20,
    'save_interval': 500
}
```

## Configuration

### Environment Setup

The algorithm supports various environment configurations:

```yaml
environment:
  name: "CartPole-v1"
  max_episode_steps: 200
  reward_threshold: 195.0
  
training:
  total_timesteps: 100000
  eval_freq: 10000
  n_eval_episodes: 10
  
ppo:
  learning_rate: 3e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  vf_coef: 0.5
  ent_coef: 0.01
```

### Network Architecture

```python
# Actor Network
actor_config = {
    'input_dim': state_dim,
    'hidden_dims': [64, 64],
    'output_dim': action_dim,
    'activation': 'tanh',
    'output_activation': 'softmax'  # for discrete actions
}

# Critic Network  
critic_config = {
    'input_dim': state_dim,
    'hidden_dims': [64, 64],
    'output_dim': 1,
    'activation': 'tanh'
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/UdulaRana/Proximal-Policy-Optimization-Algorithm.git
cd Proximal-Policy-Optimization-Algorithm
pip install -e .[dev]
pre-commit install
```

### Running Tests

```bash
pytest tests/
python -m pytest tests/ --cov=ppo
```

### Code Style

We use Black, isort, and flake8 for code formatting:

```bash
black ppo/
isort ppo/
flake8 ppo/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438.

3. Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous methods for deep reinforcement learning. In International conference on machine learning (pp. 1928-1937).

## Acknowledgments

- OpenAI for the original PPO algorithm
- The PyTorch team for the excellent deep learning framework  
- The Gym/Gymnasium community for standardized RL environments

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{ppo-implementation,
  title={Proximal Policy Optimization Algorithm Implementation},
  author={Udula Ranasinghe},
  year={2024},
  url={https://github.com/UdulaRana/Proximal-Policy-Optimization-Algorithm}
}
```

---

For more information, questions, or support, please open an issue on GitHub or contact the maintainers.
