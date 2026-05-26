#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Spring 2026, 535507 Deep Learning
# Lab7: Policy-based RL
# Task 1: A2C
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh


import random
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse
import wandb
import os
import shutil
from tqdm import tqdm
from typing import Tuple

def initialize_uniformly(layer: nn.Linear, init_w: float = 3e-3):
    """Initialize the weights and bias in [-init_w, init_w]."""
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class Actor(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialize."""
        super(Actor, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden1 = nn.Linear(in_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        # self.hidden3 = nn.Linear(256, 128)
        self.mu_layer = nn.Linear(256, out_dim)
        self.log_std_layer = nn.Linear(256, out_dim)
        self.log_std_min, self.log_std_max = -5, -1

        # initialize_uniformly(self.hidden1)
        # initialize_uniformly(self.hidden2)
        # initialize_uniformly(self.hidden3)
        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)
        #############################
        
    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))

        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to prevent NaN and ensure numerical stability
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        if deterministic:
            raw_action = mu
        else:
            raw_action = dist.rsample()

        action = torch.tanh(raw_action) * 2.0 # scale to [-2, 2]

        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(
            2.0 * (1 - torch.tanh(raw_action).pow(2) + 1e-6)
        ) # correction for Tanh squashing
        log_prob = log_prob.sum(dim=-1)

        entropy = dist.entropy().sum(dim=-1)
        #############################

        return action, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        """Initialize."""
        super(Critic, self).__init__()
        
        ############TODO#############
        # Remeber to initialize the layer weights
        self.hidden1 = nn.Linear(in_dim, 256)
        self.hidden2 = nn.Linear(256, 256)
        # self.hidden3 = nn.Linear(256, 128)
        self.value_layer = nn.Linear(256, 1)

        # initialize_uniformly(self.hidden1)
        # initialize_uniformly(self.hidden2)
        # initialize_uniformly(self.hidden3)
        initialize_uniformly(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.hidden1(state))
        x = F.relu(self.hidden2(x))
        # x = F.relu(self.hidden3(x))
        value = self.value_layer(x)
        #############################

        return value
    

class A2CAgent:
    """A2CAgent interacting with environment.

    Atribute:
        env (gym.Env): openAI Gym environment
        gamma (float): discount factor
        entropy_weight (float): rate of weighting entropy into the loss function
        device (torch.device): cpu / gpu
        actor (nn.Module): target actor model to select actions
        critic (nn.Module): critic model to predict state values
        actor_optimizer (optim.Optimizer) : optimizer of actor
        critic_optimizer (optim.Optimizer) : optimizer of critic
        transition (list): temporory storage for the recent transition
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
        seed (int): random seed
    """

    def __init__(self, env: gym.Env, args=None):
        """Initialize."""
        self.env = env
        self.gamma = args.discount_factor
        self.entropy_weight = args.entropy_weight
        self.seed = args.seed
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.num_episodes = args.num_episodes

        self.ckpt_dir = args.ckpt_dir
        self.entropy_weight_decay = args.ewd
        self.lr_annealing = args.lra
        self.ew_min = args.ew_min
        self.eval_env = gym.make("Pendulum-v1")

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # networks
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        self.actor = Actor(obs_dim, action_dim).to(self.device)
        self.critic = Critic(obs_dim).to(self.device)

        # optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # schedulers
        if self.lr_annealing:
            self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer, T_max=self.num_episodes, eta_min=self.actor_lr * 0.1
            )
            self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer, T_max=self.num_episodes, eta_min=self.critic_lr * 0.1
            )

        # transition (state, log_prob, entropy, next_state, reward, terminated)
        self.transition: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = True if args.test else False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, log_prob, entropy = self.actor(
            state,
            deterministic=self.is_test
        )
        
        if not self.is_test:
            self.transition = [state, log_prob, entropy]

        return action.cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            # return terminated rather than done for better value estimation in truncated episodes
            self.transition.extend([next_state, reward, terminated])

        return next_state, reward, done

    def update_model(self) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Update the model by gradient descent."""
        state, log_prob, entropy, next_state, reward, terminated = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        mask = 1 - int(terminated)

        ############TODO#############
        # value_loss = ?
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state_tensor).detach()
        
        # Bootstrap value from next_state
        # In Pendulum, we only set value to 0 if terminated (which never happens in normal play).
        # If truncated (time out at 200 steps), we MUST bootstrap the value of the next state.
        
        reward = reward * 0.1 # reward scaling
        target = reward + self.gamma * next_value * mask
        current_value = self.critic(state)
        value_loss = F.mse_loss(current_value, target.detach())
        #############################

        # update value
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        ############TODO#############
        # policy_loss = ?
        advantage = target.detach() - current_value.detach()
        
        policy_loss = - (log_prob * advantage + self.entropy_weight * entropy).mean()
        #############################
        # update policy
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        self.actor_optimizer.step()

        return policy_loss.item(), value_loss.item(), entropy.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        best_score = -float('inf')
        
        # Initial entropy weight if decay is enabled
        if self.entropy_weight_decay:
            init_entropy_weight = self.entropy_weight

        state, _ = self.env.reset(seed=self.seed)
        for ep in tqdm(range(1, self.num_episodes + 1)):
            actor_losses, critic_losses, scores = [], [], []

            # Get current LR from schedulers for logging
            curr_actor_lr = self.actor_optimizer.param_groups[0]['lr']
            # Linear decay for entropy weight
            if self.entropy_weight_decay:
                frac = 1.0 - (ep - 1.0) / self.num_episodes
                self.entropy_weight = max(init_entropy_weight * frac, self.ew_min)

            if ep > 1:
                state, _ = self.env.reset()
            score = 0
            done = False

            while not done:
                # self.env.render()
                action = self.select_action(state)
                next_state, reward, done = self.step(action)

                actor_loss, critic_loss, entropy = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                state = next_state
                score += reward
                step_count += 1

                # W&B logging
                wandb.log({
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "actor_lr": curr_actor_lr,
                    "entropy_weight": self.entropy_weight,
                    "entropy": entropy
                }, step=step_count) 
                # if episode ends
                if done:
                    scores.append(score)
                    tqdm.write(f"Episode {ep}: Total Reward = {score}")
                    wandb.log({"return": score}, step=step_count)
            
            # Step the schedulers at the end of each episode
            if self.lr_annealing:
                self.actor_scheduler.step()
                self.critic_scheduler.step()
            
            # Evaluation every 20 episodes
            if ep % 20 == 0:
                eval_score = self.evaluate()
                wandb.log({"eval_return": eval_score}, step=step_count)
                tqdm.write(f"--- Evaluation at Episode {ep}: Average Reward = {eval_score} ---")
                if eval_score > best_score and eval_score >= -160.0:
                    best_score = eval_score
                    path = os.path.join(self.ckpt_dir, f"step_{step_count}_score_{eval_score}.pt")
                    torch.save({
                        "actor_state_dict": self.actor.state_dict(),
                        "critic_state_dict": self.critic.state_dict(),
                    }, path)
                    tqdm.write(f"New Best Score: {best_score} at Step {step_count}. Model saved.")
                    if eval_score >= -150.0:
                        tqdm.write("=========CONGRATULATIONS!=========")

    def evaluate(self, n_episodes=20):
        """Evaluate the agent for n_episodes."""
        self.is_test = True
        total_reward = 0
        for i in range(n_episodes):
            state, _ = self.eval_env.reset(seed=i)
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
        
        self.is_test = False
        return total_reward / n_episodes

    def test(self, video_folder: str, env_step):
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
            print(f"Env Step: {env_step}, Seed: {i}, score = {score}")

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
    parser.add_argument("--wandb-run-name",     type=str,   default="pendulum-a2c-run")
    parser.add_argument("--actor-lr",           type=float, default=1e-4)
    parser.add_argument("--critic-lr",          type=float, default=1e-3)
    parser.add_argument("--discount-factor",    type=float, default=0.9)
    parser.add_argument("--num-episodes",       type=int,   default=150)
    parser.add_argument("--seed",               type=int,   default=77)
    parser.add_argument("--entropy-weight",     type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--ckpt-dir",           type=str,   default="./task1_checkpoints")
    parser.add_argument("--video-dir",          type=str,   default="./task1_videos")
    parser.add_argument("--load-ckpt",          type=str)
    parser.add_argument("--ew-min",             type=float, default=1e-3)

    parser.add_argument("--test",               action="store_true")
    parser.add_argument("--ewd",                action="store_true")
    parser.add_argument("--lra",                action="store_true")
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1")
    if args.test:
        env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    if args.test:
        agent = A2CAgent(env, args)
        ckpt = torch.load(args.load_ckpt, map_location=agent.device)
        agent.actor.load_state_dict(ckpt['actor_state_dict'])
        agent.critic.load_state_dict(ckpt['critic_state_dict'])
        env_step = args.load_ckpt.split('/')[2].split('_')[1]
        agent.test(args.video_dir, env_step)
    else:
        with_lra = "_lra" if args.lra else ""
        if args.ewd:
            run_name = f"ew_{args.entropy_weight}_ewmin_{args.ew_min}_alr_{args.actor_lr}_clr_{args.critic_lr}{with_lra}_ep_{args.num_episodes}"
        else:
            run_name = f"ew_{args.entropy_weight}_alr_{args.actor_lr}_clr_{args.critic_lr}{with_lra}_ep_{args.num_episodes}"
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=run_name, save_code=True)
        agent = A2CAgent(env, args)
        agent.train()