"""Small PPO trainer used only by the experimental local RL command."""

from typing import Any

import torch
from torch import nn

from ..envs.base import BaseEnv


class PPOTrainer:
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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

    def collect_trajectories(self, num_steps: int) -> list[dict[str, Any]]:
        if num_steps <= 0:
            raise ValueError("steps_per_iteration must be positive")
        trajectories = []
        observation, _ = self.env.reset()
        for _ in range(num_steps):
            action, log_prob, value = self.policy.get_action(observation)
            next_observation, reward, done, truncated, _ = self.env.step(action)
            terminal = done or truncated
            trajectories.append(
                {
                    "obs": observation,
                    "next_obs": next_observation,
                    "action": action,
                    "reward": reward,
                    "done": terminal,
                    "log_prob": log_prob,
                    "value": value,
                }
            )
            observation = next_observation
            if terminal:
                observation, _ = self.env.reset()
        return trajectories

    def compute_advantages(
        self, trajectories: list[dict[str, Any]], last_value: float = 0.0
    ) -> list[float]:
        advantages = []
        gae = 0.0
        for index in reversed(range(len(trajectories))):
            transition = trajectories[index]
            next_value = (
                last_value
                if index == len(trajectories) - 1
                else trajectories[index + 1]["value"]
            )
            active = 0.0 if transition["done"] else 1.0
            delta = (
                transition["reward"]
                + self.gamma * next_value * active
                - transition["value"]
            )
            gae = delta + self.gamma * self.gae_lambda * active * gae
            advantages.insert(0, gae)
        return advantages

    def train_step(
        self, trajectories: list[dict[str, Any]], advantages: list[float]
    ) -> dict[str, float]:
        if not trajectories or len(trajectories) != len(advantages):
            raise ValueError("trajectories and advantages must be non-empty and aligned")
        device = next(self.policy.parameters()).device
        observations = [item["obs"] for item in trajectories]
        actions = [item["action"] for item in trajectories]
        old_log_probs = torch.tensor(
            [item["log_prob"] for item in trajectories],
            dtype=torch.float32,
            device=device,
        )
        returns = torch.tensor(
            [adv + item["value"] for adv, item in zip(advantages, trajectories)],
            dtype=torch.float32,
            device=device,
        )
        normalized = torch.tensor(advantages, dtype=torch.float32, device=device)
        normalized = normalized - normalized.mean()
        scale = normalized.std(unbiased=False)
        if scale > 0:
            normalized = normalized / scale

        total_loss = 0.0
        updates = 0
        for _ in range(self.epochs):
            indices = torch.randperm(len(trajectories), device=device)
            for start in range(0, len(trajectories), self.batch_size):
                batch_indices = indices[start : start + self.batch_size]
                selected = batch_indices.tolist()
                log_probs, values, entropy = self.policy.evaluate(
                    [observations[index] for index in selected],
                    [actions[index] for index in selected],
                )
                ratio = torch.exp(log_probs - old_log_probs[batch_indices])
                first = ratio * normalized[batch_indices]
                second = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                ) * normalized[batch_indices]
                policy_loss = -torch.minimum(first, second).mean()
                value_loss = nn.functional.mse_loss(
                    values.reshape(-1), returns[batch_indices]
                )
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                total_loss += float(loss.detach())
                updates += 1
        return {"loss": total_loss / updates}

    def train(
        self, num_iterations: int, steps_per_iteration: int = 2048
    ) -> list[dict[str, float]]:
        results = []
        for iteration in range(num_iterations):
            trajectories = self.collect_trajectories(steps_per_iteration)
            last = trajectories[-1]
            last_value = (
                0.0 if last["done"] else self.policy.get_value(last["next_obs"])
            )
            advantages = self.compute_advantages(trajectories, last_value)
            metrics = self.train_step(trajectories, advantages)
            metrics["iteration"] = iteration
            metrics["episode_reward"] = sum(
                item["reward"] for item in trajectories
            ) / len(trajectories)
            results.append(metrics)
            print(
                f"Iteration {iteration}: loss={metrics['loss']:.4f}, "
                f"reward={metrics['episode_reward']:.2f}"
            )
        return results
