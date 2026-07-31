#!/usr/bin/env python3
"""Entry point for 'yeto rl' command."""

import argparse
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Mock environment (for testing without server) ----------
class MockCyberGymEnv:
    """Dummy environment that returns fixed rewards."""
    def __init__(self, **kwargs):
        self.step_count = 0
        self.max_steps = 10
    
    def reset(self):
        self.step_count = 0
        return {"observation": "Mock task", "action_mask": [1.0]*10}, {}
    
    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        reward = 1.0 if done else 0.1
        return {"observation": f"Step {self.step_count}"}, reward, done, False, {}
    
    def get_observation_space(self):
        from gymnasium import spaces
        return spaces.Dict({
            "observation": spaces.Text(max_length=100),
            "action_mask": spaces.Box(0, 1, shape=(10,), dtype=float)
        })
    
    def get_action_space(self):
        from gymnasium import spaces
        return spaces.Text(max_length=1000)
    
    def render(self):
        pass

# ---------- Main training function ----------
def run_rl(args):
    # Select environment
    if args.env == "mock":
        env = MockCyberGymEnv()
        action_size = None
        print("Using mock environment (no server required)")
    else:
        from yeto.rl.envs.cybergym_env import CyberGymEnv
        env = CyberGymEnv(
            task_name=args.task,
            server_host=args.server_host,
            server_port=args.server_port,
            timeout=30
        )
        action_size = None
        print(f"Using CyberGym server at {args.server_host}:{args.server_port} with task {args.task}")

    # Create policy model
    class LLMPolicy(nn.Module):
        def __init__(self, model_name: str, num_actions: int = None):
            super().__init__()
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if num_actions is None:
                num_actions = self.tokenizer.vocab_size
            self.policy_head = nn.Linear(self.llm.config.hidden_size, num_actions)
            self.value_head = nn.Linear(self.llm.config.hidden_size, 1)
            self.num_actions = num_actions
        
        def forward(self, obs):
            text = obs.get("observation", str(obs))
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = self.llm(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]
            return self.policy_head(hidden), self.value_head(hidden)
        
        def get_action(self, obs):
            with torch.no_grad():
                # FIXED: Autoregressive generation using model.generate()
                text = obs.get("observation", "")
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                generated_ids = self.llm.generate(
                    **inputs,
                    max_new_tokens=50,          # Generate 50 tokens for a more meaningful PoC
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                action_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Compute log_prob for the first token (simplified for PPO)
                logits, value = self.forward(obs)
                probs = torch.softmax(logits, dim=-1)
                # Use the first token of the generated sequence as the action index
                first_token = generated_ids[0][0].item()
                log_prob = torch.log(probs[0, first_token]).item()
                return action_text, log_prob, value.item()
        
        def get_value(self, obs):
            with torch.no_grad():
                _, value = self.forward(obs)
                return value.item()
        
        def evaluate(self, obs_list, action_list):
            log_probs_list = []
            values_list = []
            entropies_list = []
            for obs, action_text in zip(obs_list, action_list):
                logits, value = self.forward(obs)
                probs = torch.softmax(logits, dim=-1)
                # For simplicity, take the first token of the action text
                tokens = self.tokenizer.encode(action_text, add_special_tokens=False)
                if not tokens:
                    action_idx = 0
                else:
                    action_idx = tokens[0]
                log_prob = torch.log(probs[0, action_idx])
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                log_probs_list.append(log_prob)
                values_list.append(value)
                entropies_list.append(entropy)
            return (
                torch.stack(log_probs_list),
                torch.stack(values_list).squeeze(),
                torch.stack(entropies_list).mean()
            )
    
    policy = LLMPolicy(args.model, num_actions=action_size)
    
    # Use the PPO trainer
    from yeto.rl.algorithms.ppo import PPOTrainer
    trainer = PPOTrainer(
        env=env,
        policy_model=policy,
        lr=args.lr,
        gamma=args.gamma,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    
    print(f"Starting RL training on environment: {args.env}")
    print(f"Model: {args.model}")
    print(f"Budget: ${args.budget} (monitoring only)")
    
    results = trainer.train(
        num_iterations=args.iterations,
        steps_per_iteration=args.steps,
    )
    
    # FIXED: Save the entire policy (including heads)
    output_dir = args.output or f"./rl_output_{args.env}_{args.task}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LLM and tokenizer
    policy.llm.save_pretrained(output_dir)
    policy.tokenizer.save_pretrained(output_dir)
    
    # Save policy heads separately
    torch.save({
        'policy_head': policy.policy_head.state_dict(),
        'value_head': policy.value_head.state_dict(),
        'num_actions': policy.num_actions,
    }, os.path.join(output_dir, "policy_heads.pt"))
    
    print(f"Model saved to {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Run RL training with CyberGym or mock")
    parser.add_argument("--env", default="cybergym", choices=["cybergym", "mock"],
                        help="Environment: 'cybergym' (real server) or 'mock' (dummy)")
    parser.add_argument("--task", default="vulnerability_analysis", help="CyberGym task name")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="Base model name")
    parser.add_argument("--server-host", default="127.0.0.1", help="CyberGym server host")
    parser.add_argument("--server-port", type=int, default=8666, help="CyberGym server port")
    parser.add_argument("--budget", type=float, default=10.0, help="Budget in USD (monitoring)")
    parser.add_argument("--output", help="Output directory for trained model")
    parser.add_argument("--iterations", type=int, default=1, help="Number of training iterations")
    parser.add_argument("--steps", type=int, default=64, help="Steps per iteration")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epochs", type=int, default=2, help="PPO epochs per update")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    run_rl(args)

if __name__ == "__main__":
    main()