import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Dict, Any, Tuple, Optional

from ..envs.base import BaseEnv

class PPOTrainer:
    """Simple PPO implementation for LLM-based policies."""

    def __init__(
        self,
        env: BaseEnv,
        policy_model: nn.Module,
        lr: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
    ):
        self.env = env
        self.policy = policy_model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

    def collect_trajectories(self, num_steps: int) -> List[Dict]:
        """Collect trajectories by interacting with the environment."""
        trajectories = []
        obs, _ = self.env.reset()
        done = False

        for _ in range(num_steps):
            # Get action from policy
            action, log_prob, value = self.policy.get_action(obs)

            # Step environment
            next_obs, reward, done, truncated, info = self.env.step(action)

            trajectories.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "done": done or truncated,
                "log_prob": log_prob,
                "value": value,
            })

            obs = next_obs
            if done or truncated:
                obs, _ = self.env.reset()

        return trajectories

    def compute_advantages(self, trajectories: List[Dict], last_value: float = 0.0) -> List[float]:
        """Compute GAE advantages."""
        advantages = []
        gae = 0.0

        for t in reversed(range(len(trajectories))):
            if t == len(trajectories) - 1:
                next_value = last_value
            else:
                next_value = trajectories[t + 1]["value"]

            delta = trajectories[t]["reward"] + self.gamma * next_value * (1 - trajectories[t]["done"]) - trajectories[t]["value"]
            gae = delta + self.gamma * self.gae_lambda * (1 - trajectories[t]["done"]) * gae
            advantages.insert(0, gae)

        return advantages

    def train_step(self, trajectories: List[Dict], advantages: List[float]) -> Dict[str, float]:
        """Perform one PPO update step."""
        # Convert to tensors
        obs_list = [t["obs"] for t in trajectories]
        action_list = [t["action"] for t in trajectories]
        old_log_probs = torch.tensor([t["log_prob"] for t in trajectories], dtype=torch.float32)
        returns = torch.tensor([adv + t["value"] for adv, t in zip(advantages, trajectories)], dtype=torch.float32)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        total_loss = 0.0

        for _ in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(len(trajectories))

            for start in range(0, len(trajectories), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]

                # Get batch data
                batch_obs = [obs_list[i] for i in batch_indices]
                batch_actions = [action_list[i] for i in batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                # Forward pass
                log_probs, values, entropy = self.policy.evaluate(batch_obs, batch_actions)

                # PPO loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                entropy_loss = -entropy.mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                total_loss += loss.item()

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return {"loss": total_loss / (self.epochs * len(trajectories) / self.batch_size)}

    def train(self, num_iterations: int, steps_per_iteration: int = 2048) -> List[Dict]:
        """Main training loop."""
        results = []

        for iteration in range(num_iterations):
            # Collect trajectories
            trajectories = self.collect_trajectories(steps_per_iteration)

            # Compute advantages
            last_value = self.policy.get_value(trajectories[-1]["obs"]) if trajectories else 0.0
            advantages = self.compute_advantages(trajectories, last_value)

            # Train
            metrics = self.train_step(trajectories, advantages)
            metrics["iteration"] = iteration
            metrics["episode_reward"] = sum(t["reward"] for t in trajectories) / len(trajectories)
            results.append(metrics)

            print(f"Iteration {iteration}: loss={metrics['loss']:.4f}, reward={metrics['episode_reward']:.2f}")

        return results
