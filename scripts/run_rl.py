#!/usr/bin/env python3
"""Entry point for 'yeto rl' command."""

import argparse
import os
import sys
from pathlib import Path

# Add Yeto to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from yeto.rl.envs.cybergym_env import CyberGymEnv
from yeto.rl.algorithms.ppo import PPOTrainer


def get_policy_model(model_name: str, action_space_size: int):
    """Load or create a policy model (LLM + policy head)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn as nn
    
    class LLMPolicy(nn.Module):
        def __init__(self, model_name: str, num_actions: int):
            super().__init__()
            self.llm = AutoModelForCausalLM.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.policy_head = nn.Linear(self.llm.config.hidden_size, num_actions)
            self.value_head = nn.Linear(self.llm.config.hidden_size, 1)
            
        def forward(self, obs):
            # Extract text from observation dict
            text = obs.get("observation", str(obs))
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = self.llm(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]  # Last token embedding
            return self.policy_head(hidden), self.value_head(hidden)
        
        def get_action(self, obs):
            with torch.no_grad():
                logits, value = self.forward(obs)
                probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(probs, 1).item()
                log_prob = torch.log(probs[0, action])
                return action, log_prob.item(), value.item()
        
        def get_value(self, obs):
            with torch.no_grad():
                _, value = self.forward(obs)
                return value.item()
        
        def evaluate(self, obs_list, action_list):
            # Batch evaluation for training
            log_probs_list = []
            values_list = []
            entropies_list = []
            
            for obs, action in zip(obs_list, action_list):
                logits, value = self.forward(obs)
                probs = torch.softmax(logits, dim=-1)
                log_prob = torch.log(probs[0, action])
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                log_probs_list.append(log_prob)
                values_list.append(value)
                entropies_list.append(entropy)
            
            return (
                torch.stack(log_probs_list),
                torch.stack(values_list).squeeze(),
                torch.stack(entropies_list).mean()
            )
    
    return LLMPolicy(model_name, action_space_size)


def run_rl(args):
    """Run RL training with CyberGym."""
    # Create environment
    env = CyberGymEnv(task_name=args.task)
    
    # Create policy model
    action_space = env.get_action_space()
    action_size = action_space.n if hasattr(action_space, "n") else action_space.shape[0]
    policy = get_policy_model(args.model, action_size)
    
    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy_model=policy,
        lr=args.lr,
        gamma=args.gamma,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    # Train
    print(f"Starting RL training on CyberGym task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Budget: ${args.budget} (monitoring only)")
    
    results = trainer.train(
        num_iterations=args.iterations,
        steps_per_iteration=args.steps,
    )
    
    # Save model
    output_dir = args.output or f"./rl_output_{args.task}"
    os.makedirs(output_dir, exist_ok=True)
    policy.llm.save_pretrained(output_dir)
    policy.tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RL training with CyberGym")
    parser.add_argument("--env", default="cybergym", help="Environment name (default: cybergym)")
    parser.add_argument("--task", default="vulnerability_analysis", help="CyberGym task name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B", help="Base model name")
    parser.add_argument("--budget", type=float, default=10.0, help="Budget in USD (monitoring)")
    parser.add_argument("--output", help="Output directory for trained model")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--steps", type=int, default=2048, help="Steps per iteration")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epochs", type=int, default=10, help="PPO epochs per update")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    
    args = parser.parse_args()
    run_rl(args)


if __name__ == "__main__":
    main()