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
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.mu_layer = nn.Linear(512, out_dim)
        self.log_std_layer = nn.Linear(512, out_dim)

        initialize_uniformly(self.fc1)
        initialize_uniformly(self.fc2)
        initialize_uniformly(self.mu_layer)
        initialize_uniformly(self.log_std_layer)
        #############################
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, Normal]:
        """Forward method implementation."""

        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = torch.tanh(self.mu_layer(x)) * 2.0
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to prevent NaN and ensure numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
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
        self.fc1 = nn.Linear(in_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.value_layer = nn.Linear(512, 1)

        initialize_uniformly(self.fc1)
        initialize_uniformly(self.fc2)
        initialize_uniformly(self.value_layer)
        #############################

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        
        ############TODO#############
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
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
        self.n_step = args.n_step
        self.entropy_weight_decay = args.ewd
        self.lr_annealing = args.lr_annealing
        self.ew_min = args.ew_min

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
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=self.num_episodes, eta_min=self.actor_lr * 0.01
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=self.num_episodes, eta_min=self.critic_lr * 0.01
        )

        # transition (state, log_prob, next_state, reward, done)
        self.transition: list = list()
        self.transitions: list = list()

        # total steps count
        self.total_step = 0

        # mode: train / test
        self.is_test = True if args.test else False

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        state = torch.FloatTensor(state).to(self.device)
        action, dist = self.actor(state)
        selected_action = dist.mean if self.is_test else action

        if not self.is_test:
            log_prob = dist.log_prob(selected_action).sum(dim=-1)
            self.transition = [state, log_prob]

        return selected_action.clamp(-2.0, 2.0).cpu().detach().numpy()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, bool]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        
        if not self.is_test:
            # Reward scaling for better stability in Pendulum
            scaled_reward = reward / 10.0
            # Store [state, log_prob, next_state, scaled_reward, terminated]
            self.transition.extend([next_state, scaled_reward, terminated])

        return next_state, reward, terminated, truncated

    def update_model(self, next_state, terminated) -> Tuple[float, float]:
        """Update the model by gradient descent using N-step returns."""
        
        # Bootstrap value from next_state
        # In Pendulum, we only set value to 0 if terminated (which never happens in normal play).
        # If truncated (time out at 200 steps), we MUST bootstrap the value of the next state.
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        next_value = self.critic(next_state_tensor).detach()
        
        if terminated:
            next_value = torch.FloatTensor([0.0]).to(self.device)
        
        # Calculate returns backwards
        returns = []
        g = next_value
        for transition in reversed(self.transitions):
            # transition: [state, log_prob, next_state, scaled_reward, terminated]
            reward = transition[3]
            term = transition[4]
            mask = 1.0 - float(term) # Only mask if terminated
            g = reward + self.gamma * g * mask
            returns.append(g)
        returns.reverse()
        
        # Convert buffer and returns to tensors
        states = torch.stack([t[0] for t in self.transitions])
        log_probs = torch.stack([t[1] for t in self.transitions])
        returns = torch.stack(returns).detach().view(-1, 1)
        
        # Value loss
        values = self.critic(states) 
        value_loss = F.mse_loss(values, returns)
        
        # Policy loss
        advantages = returns - values.detach()
        _, dist = self.actor(states)
        entropy = dist.entropy().sum(-1).mean()
        
        policy_loss = -(log_probs * advantages.squeeze(-1)).mean() - self.entropy_weight * entropy
        
        # Update
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        loss = policy_loss + value_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        self.transitions = [] # Clear buffer
        
        return policy_loss.item(), value_loss.item()

    def train(self):
        """Train the agent."""
        self.is_test = False
        step_count = 0
        last_ckpt_step = 0
        best_score = -float('inf')
        
        # Cache losses for step-by-step logging
        actor_loss, critic_loss = 0, 0
        if self.entropy_weight_decay:
            # Initial entropy weight
            init_entropy_weight = self.entropy_weight

        state, _ = self.env.reset(seed=self.seed)
        for ep in tqdm(range(1, self.num_episodes + 1)):
            # Get current LR from schedulers for logging
            curr_actor_lr = self.actor_optimizer.param_groups[0]['lr']
            if self.entropy_weight_decay:
                # Linear decay for entropy weight
                frac = 1.0 - (ep - 1.0) / self.num_episodes
                self.entropy_weight = max(init_entropy_weight * frac, self.ew_min)

            if ep > 1:
                state, _ = self.env.reset()
            score = 0
            done = False
            self.transitions = []
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.step(action)
                done = terminated or truncated
                
                self.transitions.append(self.transition)

                state = next_state
                score += reward
                step_count += 1
                
                # N-step update
                if len(self.transitions) >= self.n_step or done:
                    actor_loss, critic_loss = self.update_model(next_state, terminated)
                
                # W&B logging
                wandb.log({
                    "step": step_count,
                    "actor loss": actor_loss,
                    "critic loss": critic_loss,
                    "actor_lr": curr_actor_lr,
                    "entropy_weight": self.entropy_weight
                }) 

                # if episode ends
                if done:
                    print(f"Episode {ep}: Total Reward = {score}")
                    wandb.log({"episode": ep, "return": score})
            
            # Step the schedulers at the end of each episode
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            
            # Evaluation every 20 episodes
            if ep % 20 == 0:
                eval_score = self.evaluate()
                wandb.log({"eval_return": eval_score})
                print(f"--- Evaluation at Episode {ep}: Average Reward = {eval_score} ---")
                if eval_score >= -150.0:
                    succes_actor_path = os.path.join(self.ckpt_dir, f"a2c_actor_pass_{step_count}.pt")
                    succes_critic_path = os.path.join(self.ckpt_dir, f"a2c_critic_pass_{step_count}.pt")
                    torch.save(self.actor.state_dict(), succes_actor_path)
                    torch.save(self.critic.state_dict(), succes_critic_path)
                    print(f"CONGRATULATIONS! Saved model for passing score at step {step_count}!")
                if eval_score > best_score:
                    best_score = eval_score
                    torch.save(self.actor.state_dict(), os.path.join(self.ckpt_dir, "a2c_actor_best.pt"))
                    torch.save(self.critic.state_dict(), os.path.join(self.ckpt_dir, "a2c_critic_best.pt"))
                    print(f"New Best Model Saved! Score: {best_score}")

            # Check for 50k step checkpoint
            if step_count // 50000 > last_ckpt_step // 50000:
                last_ckpt_step = (step_count // 50000) * 50000
                suffix = f"{last_ckpt_step // 1000}k"
                best_actor_path = os.path.join(self.ckpt_dir, "a2c_actor_best.pt")
                best_critic_path = os.path.join(self.ckpt_dir, "a2c_critic_best.pt")
                if os.path.exists(best_actor_path) and os.path.exists(best_critic_path):
                    shutil.copy(best_actor_path, os.path.join(self.ckpt_dir, f"a2c_actor_{suffix}.pt"))
                    shutil.copy(best_critic_path, os.path.join(self.ckpt_dir, f"a2c_critic_{suffix}.pt"))
                    print(f"Saved copy of best model at {suffix} steps.")

    def evaluate(self, n_episodes=20):
        """Evaluate the agent for n_episodes."""
        self.is_test = True
        total_reward = 0
        for i in range(n_episodes):
            state, _ = self.env.reset(seed=i)
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.step(action)
                done = terminated or truncated
                state = next_state
                total_reward += reward
        
        self.is_test = False
        return total_reward / n_episodes

    def test(self, video_folder: str, env_step):
        """Test the agent."""
        self.is_test = True

        tmp_env = self.env
        self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder, episode_trigger=lambda x: x == 0)

        scores = []
        for i in range(20):
            state, _ = self.env.reset(seed=i)
            done = False
            score = 0

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated = self.step(action)
                done = terminated or truncated

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
    parser.add_argument("--num-episodes",       type=int,   default=1000)
    parser.add_argument("--seed",               type=int,   default=77)
    parser.add_argument("--entropy-weight",     type=float, default=1e-2) # entropy can be disabled by setting this to 0
    parser.add_argument("--ckpt-dir",           type=str,   default="./checkpoints")
    parser.add_argument("--video-dir",          type=str,   default="./videos")
    parser.add_argument("--load-ckpt",          type=str)
    parser.add_argument("--n-step",             type=int,   default=5)
    parser.add_argument("--ew-min",             type=int,   default=1e-3)

    parser.add_argument("--test",               action="store_true")
    parser.add_argument("--ewd",                action="store_true")
    parser.add_argument("--lr-annealing",       action="store_true")
    args = parser.parse_args()
    
    # environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    seed = 77
    random.seed(seed)
    np.random.seed(seed)
    seed_torch(seed)
    
    if args.test:
        agent = A2CAgent(env, args)
        agent.actor.load_state_dict(torch.load(args.load_ckpt))
        env_step = args.load_ckpt.split('_')[-1].split('.')[0]
        agent.test(args.video_dir, env_step)
    else:
        with_lr_annealing = "w" if args.lr_annealing else "wo"
        if args.ewd:
            run_name = f"ew_{args.entropy_weight}_ewmin_{args.ew_min}_alr_{args.actor_lr}_clr_{args.critic_lr}_ep_{args.num_episodes}_nstep_{args.n_step}"
        else:
            run_name = f"woewd_ew_{args.entropy_weight}_alr_{args.actor_lr}_clr_{args.critic_lr}_ep_{args.num_episodes}_nstep_{args.n_step}"
        wandb.init(project="DLP-Lab7-A2C-Pendulum", name=run_name, save_code=True)
        agent = A2CAgent(env, args)
        agent.train()