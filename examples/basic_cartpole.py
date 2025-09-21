#!/usr/bin/env python3
"""
Basic CartPole example using PPO algorithm.

This example demonstrates the fundamental usage of PPO for training
an agent in the CartPole-v1 environment.
"""

import argparse
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# Note: In actual implementation, these would import from the ppo package
# from ppo import PPOAgent, PPOConfig


class DummyPPOAgent:
    """Placeholder PPO agent for documentation purposes."""
    
    def __init__(self, state_dim: int, action_dim: int, **kwargs):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = kwargs
        print(f"Initialized PPO Agent with state_dim={state_dim}, action_dim={action_dim}")
    
    def train(self, env, episodes: int = 1000, **kwargs) -> Dict[str, List[float]]:
        """Placeholder training method."""
        print(f"Training for {episodes} episodes...")
        # Simulate training metrics
        rewards = np.random.normal(100, 50, episodes).cumsum()
        losses = np.exp(-np.linspace(0, 5, episodes)) + 0.1 * np.random.random(episodes)
        
        return {
            'episode_rewards': rewards.tolist(),
            'actor_losses': losses.tolist(),
            'critic_losses': (losses * 0.5).tolist()
        }
    
    def test(self, env, episodes: int = 10) -> Dict[str, float]:
        """Placeholder testing method."""
        print(f"Testing for {episodes} episodes...")
        return {
            'mean_reward': 195.0,
            'std_reward': 10.0,
            'success_rate': 0.9
        }
    
    def save(self, path: str):
        """Placeholder save method."""
        print(f"Model saved to {path}")


def create_environment() -> gym.Env:
    """Create and configure the CartPole environment."""
    env = gym.make('CartPole-v1')
    return env


def plot_training_results(metrics: Dict[str, List[float]], save_path: str = None):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Actor loss
    axes[0, 1].plot(metrics['actor_losses'])
    axes[0, 1].set_title('Actor Loss')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Critic loss
    axes[1, 0].plot(metrics['critic_losses'])
    axes[1, 0].set_title('Critic Loss')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].grid(True)
    
    # Moving average of rewards
    window = 50
    if len(metrics['episode_rewards']) >= window:
        moving_avg = np.convolve(metrics['episode_rewards'], 
                               np.ones(window)/window, mode='valid')
        axes[1, 1].plot(moving_avg)
        axes[1, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    
    plt.show()


def main():
    """Main training and evaluation loop."""
    parser = argparse.ArgumentParser(description='Train PPO agent on CartPole')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                       help='Actor learning rate')
    parser.add_argument('--lr-critic', type=float, default=1e-3,
                       help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--eps-clip', type=float, default=0.2,
                       help='PPO clipping parameter')
    parser.add_argument('--k-epochs', type=int, default=4,
                       help='Number of policy update epochs')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--save-model', action='store_true',
                       help='Save trained model')
    parser.add_argument('--model-path', type=str, default='models/cartpole_ppo.pth',
                       help='Path to save/load model')
    parser.add_argument('--plot-results', action='store_true',
                       help='Plot training results')
    parser.add_argument('--test-episodes', type=int, default=10,
                       help='Number of test episodes')
    
    args = parser.parse_args()
    
    # Create environment
    print("Creating environment...")
    env = create_environment()
    
    # Get environment dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"Environment: CartPole-v1")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize PPO agent
    print("Initializing PPO agent...")
    agent = DummyPPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        eps_clip=args.eps_clip,
        k_epochs=args.k_epochs,
        batch_size=args.batch_size
    )
    
    # Train the agent
    print("Starting training...")
    training_metrics = agent.train(
        env=env,
        episodes=args.episodes,
        max_timesteps=200,
        update_timestep=2000,
        log_interval=20
    )
    
    # Save model if requested
    if args.save_model:
        import os
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        agent.save(args.model_path)
    
    # Test the trained agent
    print("Testing trained agent...")
    test_results = agent.test(env, episodes=args.test_episodes)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED")
    print("="*50)
    print(f"Final average reward: {training_metrics['episode_rewards'][-1]:.2f}")
    print(f"Test average reward: {test_results['mean_reward']:.2f} ± {test_results['std_reward']:.2f}")
    print(f"Test success rate: {test_results['success_rate']:.2%}")
    
    # Plot results if requested
    if args.plot_results:
        plot_training_results(training_metrics, 'cartpole_training_results.png')
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()