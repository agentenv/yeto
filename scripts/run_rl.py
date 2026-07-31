#!/usr/bin/env python3
"""Entry point for 'yeto rl' command."""

import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Mock environment (for testing without server) ----------
class MockCyberGymEnv:
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
        return spaces.Dict({"observation": spaces.Text(max_length=100), "action_mask": spaces.Box(0,1,shape=(10,))})
    def get_action_space(self):
        from gymnasium import spaces
        return spaces.Text(max_length=1000)
    def render(self):
        pass

# ---------- Main training function ----------
def run_rl(args):
    if args.env == "mock":
        env = MockCyberGymEnv()
        print("Using mock environment (no server required)")
    else:
        from yeto.rl.envs.cybergym_env import CyberGymEnv
        env = CyberGymEnv(
            task_name=args.task,
            server_host=args.server_host,
            server_port=args.server_port,
            timeout=30
        )
        print(f"Using CyberGym server at {args.server_host}:{args.server_port}")

    # Policy: uses the model's LM head for action log-probs, adds a value head.
    class LLMPolicy(nn.Module):
        def __init__(self, model_name: str):
            super().__init__()
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Value head (separate)
            self.value_head = nn.Linear(self.llm.config.hidden_size, 1)

        def forward(self, obs):
            text = obs.get("observation", "")
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
            outputs = self.llm(**tokens, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][:, -1, :]
            return outputs.logits, self.value_head(hidden)

        def get_action(self, obs):
            with torch.no_grad():
                prompt = obs.get("observation", "")
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                # Generate completion tokens
                generated_ids = self.llm.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=False
                )
                # The generated sequence includes the input; we need the new tokens only.
                full_ids = generated_ids.sequences[0]
                input_len = inputs.input_ids.shape[1]
                completion_ids = full_ids[input_len:]   # only the generated part
                # Decode to text
                action_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

                # Compute log-probs of the completion tokens using the model's LM head
                # We'll do a forward pass on the full sequence and get logits for the generated positions.
                # For simplicity, we take the logits from the last position? No, we need per-token log-probs.
                # We'll compute the log-prob of the entire completion as the sum (or average) of log-probs.
                # For PPO, we need a single log_prob value for the action.
                # We'll average the log-probs of the generated tokens.
                full_logits, _ = self.forward(obs)  # obs is the same as prompt
                # But that only gives logits for the last token of the input, not for the generated sequence.
                # Better: we need logits for the generated tokens from the model's output when we feed the full sequence.
                # We'll feed the full sequence (input + generated) and get logits, then shift to get next-token log-probs.
                full_input = torch.cat([inputs.input_ids, completion_ids.unsqueeze(0)], dim=1)
                with torch.no_grad():
                    outputs_full = self.llm(full_input)
                logits_full = outputs_full.logits[0]   # (seq_len, vocab)
                # For each generated token, the logits for that token come from the previous position.
                # So for the i-th generated token at position pos = input_len + i, the logits are at pos-1.
                # Actually we need logits at position (input_len + i - 1) to predict the token at input_len + i.
                # We'll compute the log-prob of each generated token.
                log_probs = []
                for i, token_id in enumerate(completion_ids):
                    pos = input_len + i - 1   # position of the last token before predicting this token
                    if pos < 0:
                        pos = 0   # should not happen
                    logits_token = logits_full[pos]   # (vocab,)
                    log_prob = torch.log_softmax(logits_token, dim=-1)[token_id].item()
                    log_probs.append(log_prob)
                avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else 0.0

                # Value
                _, value = self.forward(obs)
                return action_text, avg_log_prob, value.item()

        def get_value(self, obs):
            with torch.no_grad():
                _, value = self.forward(obs)
                return value.item()

        def evaluate(self, obs_list, action_list):
            # For training: we need log-probs of the actions (completions) and values.
            # We'll compute the average log-prob per action as above.
            log_probs = []
            values = []
            entropies = []
            for obs, action_text in zip(obs_list, action_list):
                # Compute log-prob of the action_text (the completion)
                prompt = obs.get("observation", "")
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
                completion_ids = self.tokenizer.encode(action_text, add_special_tokens=False)
                if not completion_ids:
                    # fallback: use a dummy token
                    completion_ids = [0]
                full_input = torch.cat([inputs.input_ids, torch.tensor([completion_ids], device=inputs.input_ids.device)], dim=1)
                with torch.no_grad():
                    outputs_full = self.llm(full_input)
                logits_full = outputs_full.logits[0]
                input_len = inputs.input_ids.shape[1]
                log_probs_list = []
                for i, token_id in enumerate(completion_ids):
                    pos = input_len + i - 1
                    if pos < 0:
                        pos = 0
                    logits_token = logits_full[pos]
                    log_prob = torch.log_softmax(logits_token, dim=-1)[token_id]
                    log_probs_list.append(log_prob)
                avg_log_prob = torch.stack(log_probs_list).mean() if log_probs_list else torch.tensor(0.0)
                # Value
                _, val = self.forward(obs)
                # Entropy (approximate: entropy of the policy distribution over the vocabulary from the last hidden)
                # We'll use the policy's distribution (causal LM) but we need to compute entropy over the next-token distribution.
                # For simplicity, we use the entropy of the distribution from the last position of the input.
                logits, _ = self.forward(obs)   # logits from the last token of input
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                log_probs.append(avg_log_prob)
                values.append(val.squeeze())
                entropies.append(entropy)
            return (torch.stack(log_probs), torch.stack(values), torch.stack(entropies).mean())

    policy = LLMPolicy(args.model)

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

    output_dir = args.output or f"./rl_output_{args.env}_{args.task}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the full policy (including value head)
    policy.llm.save_pretrained(output_dir)
    policy.tokenizer.save_pretrained(output_dir)
    torch.save(policy.state_dict(), os.path.join(output_dir, "policy_state_dict.pt"))

    print(f"Model saved to {output_dir}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cybergym", choices=["cybergym", "mock"])
    parser.add_argument("--task", default="vulnerability_analysis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")   # 0.5B for fast testing
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8666)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--output")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)   # hyphen consistent with CLI

    args = parser.parse_args()
    run_rl(args)

if __name__ == "__main__":
    main()