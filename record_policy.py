import os
import sys
import torch
import gymnasium as gym
import imageio
import yaml


# Add root dir to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ppo import ActorNetwork

# Load config
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    config = yaml.safe_load(f)

# Set up environment with render_mode
env = gym.make(config["env_name"], render_mode="rgb_array")
obs_dim = env.observation_space.shape
act_dim = env.action_space.n

# Load policy
policy = ActorNetwork(act_dim, obs_dim, config["learning_rate"], chkpt_dir=config["log_dir"] + "/checkpoints")
# policy_path = os.path.join("runs", "best_policy.pt")  
policy.load_checkpoint()
policy.eval()

# Recording settings
output_dir = os.path.join("videos", "ppo")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{config['env_name']}_ep0.mp4")
frames = []

# Run one episode and collect frames
obs, _ = env.reset()
done = False
total_reward = 0

for _ in range(300):
    frame = env.render()
    frames.append(frame)

    action= policy(torch.Tensor(obs).to("cuda:0")).sample().cpu().item()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    if done:
        obs, _ = env.reset()
    total_reward += reward

env.close()

# Save the video
imageio.mimsave(output_path, frames, fps=30)
print(f"🎥 Video saved to: {output_path}")
print(f"🎯 Episode reward: {total_reward}")
