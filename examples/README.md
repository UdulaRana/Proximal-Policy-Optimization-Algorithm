# Examples

This directory contains example implementations and usage scenarios for the PPO algorithm.

## Available Examples

### Basic Examples
- `basic_cartpole.py` - Simple CartPole environment training
- `basic_lunar_lander.py` - LunarLander discrete action training
- `continuous_bipedal.py` - BipedalWalker continuous action training

### Advanced Examples
- `multi_environment.py` - Training with multiple parallel environments
- `custom_environment.py` - Using PPO with custom environments
- `hyperparameter_tuning.py` - Automated hyperparameter optimization
- `transfer_learning.py` - Transfer learning between similar tasks

### Real-world Applications
- `robot_navigation.py` - Robot navigation in simulated environments
- `portfolio_optimization.py` - Financial portfolio management
- `resource_allocation.py` - Cloud resource allocation optimization

### Visualization & Analysis
- `training_analysis.py` - Analyze training performance and metrics
- `policy_visualization.py` - Visualize learned policies
- `comparative_study.py` - Compare PPO with other algorithms

## Running Examples

Each example can be run independently:

```bash
python examples/basic_cartpole.py
```

Most examples include configurable parameters:

```bash
python examples/basic_cartpole.py --episodes 2000 --lr 0.0003 --save-model
```

## Requirements

Install additional dependencies for examples:

```bash
pip install matplotlib seaborn plotly gymnasium[classic_control]
```

For specific examples:
- Robot navigation: `pip install pybullet`
- Financial examples: `pip install yfinance pandas`
- Visualization: `pip install tensorboard wandb`