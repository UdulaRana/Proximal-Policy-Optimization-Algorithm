# API Reference

## Core Classes

### PPOAgent

The main PPO agent class that implements the Proximal Policy Optimization algorithm.

```python
class PPOAgent:
    """Proximal Policy Optimization Agent.
    
    Implements the PPO algorithm with clipped surrogate objective for stable
    policy updates in reinforcement learning environments.
    """
```

#### Constructor

```python
def __init__(
    self,
    state_dim: int,
    action_dim: int,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
    gamma: float = 0.99,
    eps_clip: float = 0.2,
    k_epochs: int = 4,
    batch_size: int = 64,
    hidden_dim: int = 64,
    device: str = "auto"
) -> None:
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_dim` | `int` | Required | Dimension of the state space |
| `action_dim` | `int` | Required | Dimension of the action space |
| `lr_actor` | `float` | `3e-4` | Learning rate for the actor network |
| `lr_critic` | `float` | `1e-3` | Learning rate for the critic network |
| `gamma` | `float` | `0.99` | Discount factor for future rewards |
| `eps_clip` | `float` | `0.2` | Clipping parameter for PPO objective |
| `k_epochs` | `int` | `4` | Number of epochs for policy updates |
| `batch_size` | `int` | `64` | Batch size for training |
| `hidden_dim` | `int` | `64` | Hidden layer dimension for networks |
| `device` | `str` | `"auto"` | Device for computation ("cpu", "cuda", or "auto") |

#### Methods

##### get_action()

```python
def get_action(self, state: np.ndarray) -> Tuple[int, float]:
    """Get action from the current policy.
    
    Args:
        state: Current environment state
        
    Returns:
        Tuple of (action, action_log_probability)
    """
```

##### train()

```python
def train(
    self,
    env: gym.Env,
    episodes: int = 1000,
    max_timesteps: int = 200,
    update_timestep: int = 2000,
    log_interval: int = 20,
    save_interval: int = 500
) -> Dict[str, List[float]]:
    """Train the PPO agent.
    
    Args:
        env: OpenAI Gym environment
        episodes: Number of training episodes
        max_timesteps: Maximum timesteps per episode
        update_timestep: Timesteps between policy updates
        log_interval: Episodes between logging
        save_interval: Episodes between model saves
        
    Returns:
        Dictionary containing training metrics
    """
```

##### update()

```python
def update(self) -> Dict[str, float]:
    """Update policy and value function.
    
    Returns:
        Dictionary containing loss values and metrics
    """
```

##### save()

```python
def save(self, filepath: str) -> None:
    """Save the trained model.
    
    Args:
        filepath: Path to save the model
    """
```

##### load()

```python
def load(self, filepath: str) -> None:
    """Load a trained model.
    
    Args:
        filepath: Path to the saved model
    """
```

##### test()

```python
def test(
    self,
    env: gym.Env,
    episodes: int = 10,
    render: bool = False
) -> Dict[str, float]:
    """Test the trained agent.
    
    Args:
        env: Test environment
        episodes: Number of test episodes
        render: Whether to render the environment
        
    Returns:
        Dictionary containing test metrics
    """
```

### ContinuousPPOAgent

Extended PPO agent for continuous action spaces.

```python
class ContinuousPPOAgent(PPOAgent):
    """PPO Agent for continuous action spaces.
    
    Extends the base PPO agent to handle continuous actions using
    Gaussian policies with learnable standard deviation.
    """
```

#### Additional Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `action_std` | `float` | `0.5` | Initial standard deviation for action distribution |
| `action_std_decay_rate` | `float` | `0.05` | Decay rate for action standard deviation |
| `min_action_std` | `float` | `0.1` | Minimum action standard deviation |

### ActorNetwork

Neural network for policy approximation.

```python
class ActorNetwork(nn.Module):
    """Actor network for policy approximation.
    
    Implements a feedforward neural network that outputs action probabilities
    or action parameters for the current policy.
    """
```

#### Constructor

```python
def __init__(
    self,
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 64,
    n_layers: int = 2,
    activation: str = "tanh"
) -> None:
```

#### Methods

##### forward()

```python
def forward(self, state: torch.Tensor) -> torch.Tensor:
    """Forward pass through the network.
    
    Args:
        state: Input state tensor
        
    Returns:
        Action logits or parameters
    """
```

### CriticNetwork

Neural network for value function approximation.

```python
class CriticNetwork(nn.Module):
    """Critic network for value function approximation.
    
    Implements a feedforward neural network that estimates the value
    function V(s) for given states.
    """
```

#### Constructor

```python
def __init__(
    self,
    state_dim: int,
    hidden_dim: int = 64,
    n_layers: int = 2,
    activation: str = "tanh"
) -> None:
```

#### Methods

##### forward()

```python
def forward(self, state: torch.Tensor) -> torch.Tensor:
    """Forward pass through the network.
    
    Args:
        state: Input state tensor
        
    Returns:
        State value estimate
    """
```

### ExperienceBuffer

Buffer for storing and managing training experiences.

```python
class ExperienceBuffer:
    """Experience buffer for PPO training.
    
    Stores trajectories and computes advantages using GAE
    (Generalized Advantage Estimation).
    """
```

#### Constructor

```python
def __init__(
    self,
    buffer_size: int = 10000,
    state_dim: int = 4,
    action_dim: int = 2,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> None:
```

#### Methods

##### add()

```python
def add(
    self,
    state: np.ndarray,
    action: int,
    reward: float,
    next_state: np.ndarray,
    done: bool,
    log_prob: float,
    value: float
) -> None:
    """Add experience to buffer.
    
    Args:
        state: Current state
        action: Action taken
        reward: Reward received
        next_state: Next state
        done: Whether episode ended
        log_prob: Log probability of action
        value: State value estimate
    """
```

##### compute_advantages()

```python
def compute_advantages(self, next_value: float = 0.0) -> None:
    """Compute advantages using GAE.
    
    Args:
        next_value: Value of the next state (for bootstrapping)
    """
```

##### get_batch()

```python
def get_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
    """Get a random batch of experiences.
    
    Args:
        batch_size: Size of the batch
        
    Returns:
        Dictionary containing batched experiences
    """
```

##### clear()

```python
def clear(self) -> None:
    """Clear the buffer."""
```

## Utility Functions

### Environment Utilities

```python
def make_env(env_name: str, **kwargs) -> gym.Env:
    """Create and configure environment.
    
    Args:
        env_name: Name of the environment
        **kwargs: Additional environment parameters
        
    Returns:
        Configured environment
    """

def wrap_env(env: gym.Env, **kwargs) -> gym.Env:
    """Apply common environment wrappers.
    
    Args:
        env: Base environment
        **kwargs: Wrapper configuration
        
    Returns:
        Wrapped environment
    """
```

### Training Utilities

```python
def compute_gae(
    rewards: List[float],
    values: List[float],
    next_value: float,
    gamma: float = 0.99,
    lambda_gae: float = 0.95
) -> List[float]:
    """Compute Generalized Advantage Estimation.
    
    Args:
        rewards: List of rewards
        values: List of value estimates
        next_value: Next state value
        gamma: Discount factor
        lambda_gae: GAE parameter
        
    Returns:
        List of advantage estimates
    """

def normalize_advantages(advantages: np.ndarray) -> np.ndarray:
    """Normalize advantages to have zero mean and unit variance.
    
    Args:
        advantages: Raw advantage estimates
        
    Returns:
        Normalized advantages
    """
```

### Evaluation Utilities

```python
def evaluate_agent(
    agent: PPOAgent,
    env: gym.Env,
    n_episodes: int = 10,
    render: bool = False
) -> Dict[str, float]:
    """Evaluate agent performance.
    
    Args:
        agent: Trained PPO agent
        env: Evaluation environment
        n_episodes: Number of evaluation episodes
        render: Whether to render episodes
        
    Returns:
        Dictionary containing evaluation metrics
    """

def plot_training_curves(
    metrics: Dict[str, List[float]],
    save_path: str = None
) -> None:
    """Plot training curves.
    
    Args:
        metrics: Training metrics
        save_path: Path to save plot
    """
```

## Configuration Classes

### PPOConfig

```python
@dataclass
class PPOConfig:
    """Configuration class for PPO agent."""
    
    # Environment settings
    env_name: str = "CartPole-v1"
    max_episode_steps: int = 200
    
    # Network architecture
    hidden_dim: int = 64
    n_layers: int = 2
    activation: str = "tanh"
    
    # Training parameters
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    lambda_gae: float = 0.95
    
    # PPO specific
    eps_clip: float = 0.2
    k_epochs: int = 4
    batch_size: int = 64
    
    # Training loop
    max_episodes: int = 1000
    update_timestep: int = 2000
    log_interval: int = 20
    save_interval: int = 500
    
    # Device settings
    device: str = "auto"
    seed: int = 42
```

## Examples

### Basic Usage

```python
import gym
from ppo import PPOAgent

# Create environment
env = gym.make('CartPole-v1')

# Initialize agent
agent = PPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n
)

# Train agent
metrics = agent.train(env, episodes=1000)

# Test agent
test_results = agent.test(env, episodes=10)
print(f"Average test reward: {test_results['mean_reward']:.2f}")
```

### Advanced Configuration

```python
from ppo import PPOAgent, PPOConfig

# Create custom configuration
config = PPOConfig(
    env_name="LunarLander-v2",
    hidden_dim=128,
    lr_actor=1e-4,
    eps_clip=0.1,
    k_epochs=8,
    max_episodes=2000
)

# Initialize agent with config
agent = PPOAgent.from_config(config)

# Train with custom environment
env = gym.make(config.env_name)
agent.train(env)
```

### Continuous Actions

```python
from ppo import ContinuousPPOAgent

env = gym.make('BipedalWalker-v3')

agent = ContinuousPPOAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    action_std=0.3,
    lr_actor=1e-4
)

agent.train(env, episodes=5000)
```

## Error Handling

The API includes comprehensive error handling for common issues:

### Invalid Parameters

```python
try:
    agent = PPOAgent(state_dim=-1, action_dim=2)
except ValueError as e:
    print(f"Invalid parameters: {e}")
```

### Environment Compatibility

```python
try:
    agent.train(incompatible_env)
except EnvironmentError as e:
    print(f"Environment compatibility issue: {e}")
```

### Device Errors

```python
try:
    agent = PPOAgent(device="cuda")
except RuntimeError as e:
    print(f"CUDA not available: {e}")
```

## Performance Notes

- Use GPU acceleration when available for better performance
- Batch processing is automatically handled for efficiency
- Memory usage scales with buffer size and network complexity
- Consider using parallel environments for faster training

For more detailed examples and tutorials, see the `examples/` directory.