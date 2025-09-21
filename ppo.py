import os

import torch
import torch.nn as nn


import torch.optim as optim
import numpy as np


class PPOMemory:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.vals = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + self.batch_size] for i in batch_start]

        # TODO:  Implement batching logic
        # for start in batch_start:
        #     end = start + self.batch_size
        #     if end > n_states:
        #         end = n_states
        #         Warning(f'Last batch size is smaller than {self.batch_size}.')
        #     batches = [indices[start:end]]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches,
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha,
        fc1_dims=256,
        fc2_dims=256,
        chkpt_dir="runs/ppo",
    ):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1),
        )
        self.chkpt_file = os.path.join(chkpt_dir, "ppo_actor")
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        probabilities = self.actor(state)
        dist = torch.distributions.Categorical(probabilities)
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file, map_location=self.device))


class CriticNetwork(nn.Module):
    def __init__(
        self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir="runs/ppo"
    ):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1),
        )
        self.chkpt_file = os.path.join(chkpt_dir, "ppo_critic")
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file, map_location=self.device))


class Agent:
    def __init__(
        self,
        n_actions,
        input_dims,
        alpha=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        n_epochs=10,
        batch_size=64,
        chkpt_dir="runs/ppo",
    ):
        """Initialize the PPO agent with actor and critic networks.
        Args:
            n_actions (int): Number of actions in the environment.
            input_dims (tuple): Dimensions of the input state.
            alpha (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            gae_lambda (float): Lambda for Generalized Advantage Estimation.
            clip_epsilon (float): Clipping parameter for PPO.
            entropy_coeff (float): Coefficient for entropy bonus.
            value_coeff (float): Coefficient for value loss.
            n_epochs (int): Number of epochs to train per update.
            batch_size (int): Size of each batch for training.
            chkpt_dir (str): Directory to save model checkpoints.
        """
        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims, alpha=alpha, chkpt_dir=chkpt_dir)
        self.memory = PPOMemory(batch_size)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.chkpt_dir = chkpt_dir

    def remember(self, state, action, prob, vals, reward, done):
        """Store the experience in memory.
        Args:
            state (np.array): Current state of the environment.
            action (int): Action taken by the agent.
            prob (float): Probability of the action taken.
            vals (float): Value estimate from the critic.
            reward (float): Reward received from the environment.
            done (bool): Whether the episode has ended.
        """
        self.memory.store_memory(state, action, prob, vals, reward, done)

    def save_models(self):
        """Save the actor and critic models."""
        print("[INFO] saving models")
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        """Load the actor and critic models."""
        print("[INFO] loading models")
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        """Choose an action based on the current observation.
        Args:
            observation (np.array): Current state of the environment.
        Returns:
            action (int): Action chosen by the agent.
            log_prob (float): Log probability of the action.
            value (float): Value estimate from the critic.
        """
        state = (
            torch.tensor(observation, dtype=torch.float)
            .unsqueeze(0)
            .to(self.actor.device)
        )
        dist = self.actor(state)
        action = dist.sample()

        log_prob = dist.log_prob(action)
        # action = action.squeeze(0)
        value = self.critic(state).squeeze(0)


        return action.item(), log_prob.item(), value.item()

    def learn(self):
        """Perform a learning step using the collected experiences."""
        for _ in range(self.n_epochs):
            (
                states_arr,
                actions_arr,
                probs_arr,
                vals_arr,
                rewards_arr,
                dones_arr,
                batches,
            ) = self.memory.generate_batches()

            advantages = np.zeros_like(rewards_arr, dtype=np.float32)

            # TODO:  use decue to fo the following computation efficiently
            for k in range(len(rewards_arr) - 1):
                discount = 1.0
                A_t = 0.0
                for t in range(k, len(rewards_arr) - 1):
                    A_t += discount*(
                        rewards_arr[t]
                        + self.gamma * vals_arr[t + 1] * (1 - int(dones_arr[t]))
                        - vals_arr[t]
                    )
                    discount *= self.gamma * self.gae_lambda
                advantages[k] = A_t

            advantages = torch.tensor(advantages, dtype=torch.float32).to(
                self.actor.device
            )
            values = torch.tensor(vals_arr, dtype=torch.float32).to(self.actor.device)

            for batch in batches:
                states = torch.tensor(states_arr[batch], dtype=torch.float32).to(
                    self.actor.device
                )
                actions = torch.tensor(actions_arr[batch], dtype=torch.int64).to(
                    self.actor.device
                )
                old_probs = torch.tensor(probs_arr[batch], dtype=torch.float32).to(
                    self.actor.device
                )
                values_batch = values[batch]

                dist = self.actor(states)
                new_probs = dist.log_prob(actions)

                ratios = torch.exp(new_probs - old_probs)
                advantages_batch = advantages[batch]

                # print("states shape", states.shape, "ratio shape:", ratios.shape, "actions shape:", actions.shape)

                surrogate1 = ratios * advantages_batch
                surrogate2 = (
                    torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages_batch
                )

                actor_loss = -torch.min(surrogate1, surrogate2).mean()

                # Compute critic loss
                critic_value = self.critic(states).squeeze(0)
                returns = advantages_batch + values_batch
                critic_loss = ((returns - critic_value) ** 2).mean()

                entropy_loss = dist.entropy().mean()

                total_loss = (
                    actor_loss
                    + self.value_coeff * critic_loss
                    - self.entropy_coeff * entropy_loss
                )

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        # Clear memory after learning
        self.memory.clear_memory()
