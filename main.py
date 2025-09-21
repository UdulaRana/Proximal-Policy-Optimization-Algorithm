import os
import yaml
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from ppo import Agent
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # Load configuration
    with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
        config = yaml.safe_load(f)

    # Initialize environment
    env = gym.make(config["env_name"])
    agent = Agent(n_actions=env.action_space.n,
        input_dims=env.observation_space.shape,
        alpha=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=config["clip_epsilon"],
        entropy_coeff=config["entropy_coeff"],
        value_coeff=config["value_coeff"],
        n_epochs=config["num_epoches"],
        batch_size=config["batch_size"],
        chkpt_dir=config["log_dir"]+ "/checkpoints",
    )
    writer = SummaryWriter(log_dir=config["log_dir"])

    best_score = 0.0  # Initialize best score
    reward_history = []
    avg_reward = 0
    n_steps = 0
    N = 20


    # Training loop
    for episode in range(config["num_episodes"]):
        obs, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, log_prob, val = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            n_steps += 1
            agent.remember(obs, action, log_prob, val, reward, done)
            obs = next_obs
            total_reward += reward

            if n_steps% N == 0:
                agent.learn()
            obs = next_obs
            # Logging
        writer.add_scalar("Return", total_reward, episode)
        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        if avg_reward > (best_score + 5): # +5 to add buffer to the save function
            best_score = avg_reward
            agent.save_models()
        print(f"Episode {episode + 1}: Average Reward: {avg_reward}")
    # pritn reward_history
    plt.figure()
    plt.plot(reward_history)
    plt.xlabel('Episode') 
    plt.show()  
    env.close()
    writer.close()