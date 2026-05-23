#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 3: PPO-Clip
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import os
import shutil
import random
from collections import deque
from typing import Deque, List, Tuple

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
from tqdm import tqdm

def init_layer_uniform(layer: nn.Linear, init_w: float = 3e-3) -> nn.Linear:
    """Init uniform parameters on the single layer."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)

    return layer


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        log_std_min: int = -20,
        log_std_max: int = 0,
    ):
        """Initialize."""
        super(Actor, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden1 = nn.Linear(in_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.mu_layer = nn.Linear(256, out_dim)
        self.log_std_layer = nn.Linear(256, out_dim)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

        init_layer_uniform(self.mu_layer)
        init_layer_uniform(self.log_std_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.tanh(self.hidden1(state))
        x = F.tanh(self.hidden2(x))

        mu = torch.tanh(self.mu_layer(x))
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to prevent NaN and ensure numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std) + 1e-5
        
        dist = Normal(mu, std)
        action = dist.sample()
        #############################

        return action, dist


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()

        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden1 = nn.Linear(in_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        self.value_layer = nn.Linear(256, 1)

        init_layer_uniform(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.tanh(self.hidden1(state))
        x = F.tanh(self.hidden2(x))
        value = self.value_layer(x)
        #############################

        return value
    
def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float) -> List:
    """Compute gae."""

    ############TODO#############
    values = values + [next_value]
    gae = 0
    returns = []
    
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    gae_returns = returns
    #############################
    return gae_returns

# PPO updates the model several times(update_epoch) using the stacked memory. 
# By ppo_iter function, it can yield the samples of stacked memory by interacting a environment.
def ppo_iter(
    update_epoch: int,
    mini_batch_size: int,
    states: torch.Tensor,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Get mini-batches."""
    batch_size = states.size(0)
    for _ in range(update_epoch):
        for _ in range(batch_size // mini_batch_size):
            # 本來是隨機取樣可重複，為了確保每個樣本都被訓練到，改成不重複的隨機取樣
            ids = np.random.permutation(batch_size)
            for i in range(0, batch_size, mini_batch_size):
                if i + mini_batch_size > batch_size:
                    break
                rand_ids = ids[i : i + mini_batch_size]
            yield states[rand_ids, :], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]

class PPOAgent:
    """PPO Agent.
    Attributes:
        env (gym.Env): Gym env for training
        gamma (float): discount factor
        tau (float): lambda of generalized advantage estimation (GAE)
        batch_size (int): batch size for sampling
        epsilon (float): amount of clipping surrogate objective
        update_epoch (int): the number of update
        rollout_len (int): the number of rollout
        entropy_weight (float): rate of weighting entropy into the loss function
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        transition (list): temporory storage for the recent transition
        device (torch.device): cpu / gpu
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.tau = args.tau
        self.batch_size = args.batch_size
        self.epsilon = args.epsilon
        self.num_episodes = args.num_episodes
        self.rollout_len = args.rollout_len
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.update_epoch = args.update_epoch
        self.ckpt_dir = args.ckpt_dir
        self.eval_env = gym.make("Walker2d-v5", render_mode="rgb_array")

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # networks
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.actor = Actor(self.obs_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # memory for training
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []

        # total steps count
        self.total_step = 1

        # mode: train / test
        self.is_test = False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            value = self.critic(state)
            self.states.append(state)
            self.actions.append(selected_action)
            self.values.append(value)
            self.log_probs.append(dist.log_prob(selected_action))

        return selected_action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        next_state = np.reshape(next_state, (1, -1)).astype(np.float64)
        reward = np.reshape(reward, (1, -1)).astype(np.float64)
        done = np.reshape(done, (1, -1))

        if not self.is_test:
            self.rewards.append(torch.FloatTensor(reward).to(self.device))
            self.masks.append(torch.FloatTensor(1 - done).to(self.device))

        return next_state, reward, done

    def update_model(self, next_state: np.ndarray) -> Tuple[float, float, float]:
        """Update the model by gradient descent."""
        next_state = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state)

        returns = compute_gae(
            next_value,
            self.rewards,
            self.masks,
            self.values,
            self.gamma,
            self.tau,
        )

        states = torch.cat(self.states).view(-1, self.obs_dim)
        actions = torch.cat(self.actions)
        returns = torch.cat(returns).detach()
        values = torch.cat(self.values).detach()
        log_probs = torch.cat(self.log_probs).detach()
        advantages = returns - values

        actor_losses, critic_losses = [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(
            update_epoch=self.update_epoch,
            mini_batch_size=self.batch_size,
            states=states,
            actions=actions,
            values=values,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
        ):
            # calculate ratios
            _, dist = self.actor(state)
            log_prob = dist.log_prob(action)
            ratio = (log_prob - old_log_prob).exp()

            # actor_loss
            ############TODO#############
            # actor_loss = ?
            entropy = dist.entropy().mean()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * adv
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_weight * entropy
            #############################

            # critic_loss
            ############TODO#############
            # critic_loss = ?
            critic_loss = F.mse_loss(self.critic(state), return_)
            #############################
            
            # train critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            # train actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)

        return actor_loss, critic_loss, entropy

    def train(self):
        """Train the PPO agent."""
        self.is_test = False

        state, _ = self.env.reset(seed=self.seed)
        state = np.expand_dims(state, axis=0)

        actor_losses, critic_losses = [], []
        scores = []
        score = 0
        episode_count = 0
        best_score = -float('inf')
        
        snapshot_steps = [1000000, 1500000, 2000000, 2500000, 3000000]
        saved_snapshots = set()

        for ep in tqdm(range(1, self.num_episodes + 1)):
            score = 0
            for _ in range(self.rollout_len):
                self.total_step += 1
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward[0][0]

                # if episode ends
                if done[0][0]:
                    episode_count += 1
                    state, _ = self.env.reset()
                    state = np.expand_dims(state, axis=0)
                    scores.append(score)
                    
                    wandb.log({
                        "return": score, 
                        "episode": episode_count
                    }, step=self.total_step)
                    
                    print(f"Episode {episode_count}: Total Reward = {score}")
                    score = 0

                # Check for snapshots
                for s_step in snapshot_steps:
                    if self.total_step >= s_step and s_step not in saved_snapshots:
                        best_model_path = os.path.join(self.ckpt_dir, "task3_best.pt")
                        snapshot_path = os.path.join(self.ckpt_dir, f"task3_ppo_{s_step}.pt")
                        if os.path.exists(best_model_path):
                            shutil.copy(best_model_path, snapshot_path)
                            print(f"Copied best model to {snapshot_path}")
                        else:
                            self.save_model(f"task3_ppo_{s_step}.pt")
                            print(f"Saved current model to {snapshot_path}")
                        saved_snapshots.add(s_step)

            actor_loss, critic_loss, entropy = self.update_model(next_state)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            wandb.log({
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
            }, step=self.total_step)

            eval_score = self.evaluate()
            wandb.log({"eval_score": eval_score}, step=self.total_step)
            print(f"--- Evaluation at Step {self.total_step}: Average Reward = {eval_score} ---")

            if eval_score > best_score:
                best_score = eval_score
                self.save_model("task3_best.pt")
                print(f"New best score: {best_score} at Step {self.total_step}. Models saved.")
                if eval_score >= 2500.0:
                    print("=========CONGRATULATIONS!=========")

        # termination
        self.env.close()

    def evaluate(self, n_episodes=20):
        """Evaluate the agent."""
        self.is_test = True
        total_reward = 0
        for i in range(n_episodes):
            state, _ = self.eval_env.reset(seed=i)
            state = np.expand_dims(state, axis=0)
            done = False
            while not done:
                action = self.select_action(state)
                action = action.reshape(self.action_dim,)

                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                
                state = np.expand_dims(next_state, axis=0)
                total_reward += reward
        
        self.is_test = False
        return total_reward / n_episodes

    def save_model(self, filename: str):
        """Save model parameters."""
        path = os.path.join(self.ckpt_dir, filename)
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
        }, path)

    def test(self, video_folder: str):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)

        scores = []
        for i in range(20):
            state, _ = self.env.reset(seed=i)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                state = next_state
                score += reward
            
            scores.append(score)
            print(f"Episode {i}: score = {score}")

        print("Average score: ", np.mean(scores))
        self.env.close()

        self.env = tmp_env
 
def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-run-name", type=str, default="walker-ppo-run")
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--num-episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--entropy-weight", type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--tau", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--rollout-len", type=int, default=2000)
    parser.add_argument("--update-epoch", type=int, default=10)
    parser.add_argument("--ckpt-dir", type=str, default="./task3_checkpoints")
    args = parser.parse_args()
 
    # environment
    env = gym.make("Walker2d-v5", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    wandb.init(project="DLP-Lab7-PPO-Walker", name=args.wandb_run_name, save_code=True)
    
    agent = PPOAgent(env, args)
    agent.train()