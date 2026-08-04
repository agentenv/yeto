"""Experimental single-process reinforcement-learning runner for ``yeto rl``."""

import argparse
import os

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class MockCyberGymEnv:
    def __init__(self):
        self.step_count = 0
        self.max_steps = 10

    def reset(self):
        self.step_count = 0
        return {"observation": "Mock task", "action_mask": [1.0] * 10}, {}

    def step(self, action):
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return (
            {"observation": f"Step {self.step_count}"},
            1.0 if done else 0.1,
            done,
            False,
            {},
        )

    def get_observation_space(self):
        from gymnasium import spaces

        return spaces.Dict(
            {
                "observation": spaces.Text(max_length=100),
                "action_mask": spaces.Box(0, 1, shape=(10,)),
            }
        )

    def get_action_space(self):
        from gymnasium import spaces

        return spaces.Text(max_length=1000)

    def render(self) -> None:
        return None


class LLMPolicy(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.value_head = nn.Linear(self.llm.config.hidden_size, 1)

    def _prompt_tokens(self, observation):
        tokens = self.tokenizer(
            observation.get("observation", ""),
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        return tokens.to(next(self.llm.parameters()).device)

    def forward(self, observation):
        tokens = self._prompt_tokens(observation)
        outputs = self.llm(**tokens, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1, :]
        return outputs.logits, self.value_head(hidden)

    def get_action(self, observation):
        with torch.no_grad():
            inputs = self._prompt_tokens(observation)
            generated = self.llm.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=False,
            )
            input_len = inputs.input_ids.shape[1]
            completion_ids = generated.sequences[0][input_len:]
            action = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )
            full_input = torch.cat(
                [inputs.input_ids, completion_ids.unsqueeze(0)], dim=1
            )
            logits = self.llm(full_input).logits[0]
            log_probs = [
                torch.log_softmax(logits[input_len + index - 1], dim=-1)[token]
                for index, token in enumerate(completion_ids)
            ]
            average_log_prob = (
                float(torch.stack(log_probs).mean()) if log_probs else 0.0
            )
            _, value = self.forward(observation)
            return action, average_log_prob, float(value.squeeze())

    def get_value(self, observation):
        with torch.no_grad():
            _, value = self.forward(observation)
            return float(value.squeeze())

    def evaluate(self, observations, actions):
        log_probs = []
        values = []
        entropies = []
        for observation, action in zip(observations, actions):
            inputs = self._prompt_tokens(observation)
            completion_ids = self.tokenizer.encode(
                action, add_special_tokens=False
            ) or [self.tokenizer.eos_token_id or 0]
            completion = torch.tensor(
                [completion_ids], device=inputs.input_ids.device
            )
            full_input = torch.cat([inputs.input_ids, completion], dim=1)
            outputs = self.llm(full_input, output_hidden_states=True)
            input_len = inputs.input_ids.shape[1]
            token_log_probs = [
                torch.log_softmax(
                    outputs.logits[0, input_len + index - 1], dim=-1
                )[token]
                for index, token in enumerate(completion_ids)
            ]
            log_probs.append(torch.stack(token_log_probs).mean())
            prompt_hidden = outputs.hidden_states[-1][:, input_len - 1, :]
            values.append(self.value_head(prompt_hidden).squeeze())
            probabilities = torch.softmax(
                outputs.logits[:, input_len - 1, :], dim=-1
            )
            entropies.append(
                -(probabilities * torch.log(probabilities + 1e-8)).sum()
            )
        return (
            torch.stack(log_probs),
            torch.stack(values),
            torch.stack(entropies).mean(),
        )


def run_rl(args):
    if args.env == "mock":
        env = MockCyberGymEnv()
        print("Using mock environment (no server required)")
    else:
        try:
            from .envs.cybergym_env import CyberGymEnv
        except ImportError as exc:
            raise RuntimeError(
                "CyberGym local RL requires `pip install -e '.[local-rl]'`"
            ) from exc
        env = CyberGymEnv(
            task_name=args.task,
            server_host=args.server_host,
            server_port=args.server_port,
            timeout=30,
        )
        print(f"Using CyberGym server at {args.server_host}:{args.server_port}")

    policy = LLMPolicy(args.model)
    from .algorithms.ppo import PPOTrainer

    trainer = PPOTrainer(
        env=env,
        policy_model=policy,
        lr=args.lr,
        gamma=args.gamma,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"Starting local RL training on environment: {args.env}")
    print(f"Model: {args.model}")
    print(f"Budget: ${args.budget} (monitoring only)")
    results = trainer.train(args.iterations, args.steps)

    output_dir = args.output or f"./rl_output_{args.env}_{args.task}"
    os.makedirs(output_dir, exist_ok=True)
    policy.llm.save_pretrained(output_dir)
    policy.tokenizer.save_pretrained(output_dir)
    torch.save(
        policy.state_dict(), os.path.join(output_dir, "policy_state_dict.pt")
    )
    print(f"Model saved to {output_dir}")
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="cybergym", choices=["cybergym", "mock"])
    parser.add_argument("--task", default="vulnerability_analysis")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--server-host", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=8666)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--output")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    return run_rl(parser.parse_args(argv))
