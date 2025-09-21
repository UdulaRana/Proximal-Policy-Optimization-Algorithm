# Getting Started with PPO

This guide will help you get up and running with the Proximal Policy Optimization algorithm implementation.

## Quick Installation

### Option 1: Basic Installation

```bash
git clone https://github.com/UdulaRana/Proximal-Policy-Optimization-Algorithm.git
cd Proximal-Policy-Optimization-Algorithm
pip install -r requirements.txt
```

### Option 2: Development Installation

```bash
git clone https://github.com/UdulaRana/Proximal-Policy-Optimization-Algorithm.git
cd Proximal-Policy-Optimization-Algorithm
pip install -e .[dev]
```

## Your First PPO Agent

Let's create a simple PPO agent for the CartPole environment:

```python
import gymnasium as gym
from ppo import PPOAgent

# Create environment
env = gym.make('CartPole-v1')

# Initialize agent
agent = PPOAgent(
    state_dim=4,     # CartPole observation space
    action_dim=2,    # CartPole action space (left/right)
    lr_actor=3e-4,   # Learning rate for policy network
    lr_critic=1e-3,  # Learning rate for value network
    gamma=0.99,      # Discount factor
    eps_clip=0.2     # PPO clipping parameter
)

# Train the agent
metrics = agent.train(env, episodes=1000)

# Test the trained agent
results = agent.test(env, episodes=10)
print(f"Average reward: {results['mean_reward']}")
```

## Understanding the Components

### 1. The Agent

The `PPOAgent` is the main class that implements the PPO algorithm:

- **Actor Network**: Learns the policy π(a|s)
- **Critic Network**: Learns the value function V(s)
- **Experience Buffer**: Stores trajectories for batch updates
- **Optimizer**: Updates network parameters

### 2. Training Process

The training follows this cycle:

1. **Collect Experience**: Agent interacts with environment
2. **Compute Advantages**: Calculate advantage estimates using GAE
3. **Update Policy**: Optimize the clipped PPO objective
4. **Update Value Function**: Minimize value prediction error

### 3. Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `lr_actor` | Policy network learning rate | 1e-4 to 3e-4 |
| `lr_critic` | Value network learning rate | 1e-3 to 3e-3 |
| `gamma` | Discount factor | 0.95 to 0.99 |
| `eps_clip` | PPO clipping parameter | 0.1 to 0.3 |
| `k_epochs` | Update epochs per iteration | 3 to 10 |

## Common Use Cases

### Discrete Action Spaces

For environments with discrete actions (like CartPole, Atari games):

```python
from ppo import PPOAgent

agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,  # Number of discrete actions
    eps_clip=0.2
)
```

### Continuous Action Spaces

For environments with continuous actions (like robotics, control):

```python
from ppo import ContinuousPPOAgent

agent = ContinuousPPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],  # Dimension of action vector
    action_std=0.3,  # Initial exploration noise
    eps_clip=0.2
)
```

### Custom Network Architecture

You can customize the neural network architecture:

```python
agent = PPOAgent(
    state_dim=4,
    action_dim=2,
    hidden_dim=128,    # Wider networks
    n_layers=3,        # Deeper networks
    activation='relu'  # Different activation functions
)
```

## Training Tips

### 1. Monitor Training Progress

```python
import matplotlib.pyplot as plt

# Train with metrics collection
metrics = agent.train(env, episodes=1000)

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(metrics['episode_rewards'])
plt.title('Episode Rewards')
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(1, 3, 2)
plt.plot(metrics['actor_losses'])
plt.title('Actor Loss')
plt.xlabel('Update')
plt.ylabel('Loss')

plt.subplot(1, 3, 3)
plt.plot(metrics['critic_losses'])
plt.title('Critic Loss')
plt.xlabel('Update')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
```

### 2. Save and Load Models

```python
# Save trained model
agent.save('trained_models/cartpole_ppo.pth')

# Load model later
new_agent = PPOAgent(state_dim=4, action_dim=2)
new_agent.load('trained_models/cartpole_ppo.pth')
```

### 3. Hyperparameter Tuning

Start with these guidelines:

**For Stable Training:**
- Use smaller learning rates (1e-4)
- Smaller clipping (0.1)
- More training epochs (8-10)

**For Faster Training:**
- Larger learning rates (3e-4)
- Larger clipping (0.3)
- Fewer training epochs (3-5)

**For Better Sample Efficiency:**
- Higher GAE lambda (0.95-0.99)
- Larger batch sizes (128-256)
- More update steps per episode

## Environment Setup

### Popular Environments

```python
# Classic Control
env = gym.make('CartPole-v1')        # Balance pole
env = gym.make('Acrobot-v1')         # Swing up pendulum
env = gym.make('MountainCar-v0')     # Climb mountain

# Box2D
env = gym.make('LunarLander-v2')     # Land spacecraft
env = gym.make('BipedalWalker-v3')   # Walk forward

# Atari (requires additional setup)
env = gym.make('PongNoFrameskip-v4') # Play Pong
```

### Environment Wrappers

For better performance, you might want to use environment wrappers:

```python
from ppo.environments import wrap_env

# Apply common wrappers
env = wrap_env(
    gym.make('CartPole-v1'),
    normalize_observations=True,
    normalize_rewards=True,
    clip_rewards=True
)
```

## Debugging Common Issues

### 1. Training Not Converging

**Symptoms**: Reward stays flat or decreases
**Solutions**:
- Reduce learning rates
- Increase batch size
- Check reward scaling
- Verify environment setup

### 2. Training Unstable

**Symptoms**: Large reward fluctuations
**Solutions**:
- Reduce clipping parameter
- Use reward normalization
- Decrease update frequency
- Add gradient clipping

### 3. Poor Final Performance

**Symptoms**: Low test rewards after training
**Solutions**:
- Train for more episodes
- Tune hyperparameters
- Check for overfitting
- Verify evaluation setup

## Next Steps

1. **Try Different Environments**: Experiment with various Gym environments
2. **Custom Environments**: Create your own environments
3. **Advanced Features**: Explore multi-agent PPO, hierarchical RL
4. **Performance Optimization**: Use GPU acceleration, parallel environments
5. **Research Extensions**: Implement PPO variants and improvements

## Getting Help

- Check the [API Reference](api.md) for detailed documentation
- Review [examples/](../examples/) for more complex use cases
- Read the [Algorithm Documentation](algorithm.md) for theoretical background
- Open an issue on GitHub for bugs or feature requests

## Additional Resources

### Tutorials
- [PPO Paper](https://arxiv.org/abs/1707.06347) - Original research paper
- [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html) - Detailed explanation
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) - Reference implementation

### Community
- [r/MachineLearning](https://reddit.com/r/MachineLearning) - Discussions
- [RL Discord](https://discord.gg/xhfNqQv) - Real-time help
- [Papers with Code](https://paperswithcode.com/method/ppo) - Latest research

Now you're ready to start training your own PPO agents! 🚀