"""CPU-only behavior tests for the experimental local PPO loop."""

from types import SimpleNamespace

import torch

from yeto.rl.algorithms.ppo import PPOTrainer
from yeto.rl.run import LLMPolicy


class TinyEnv:
    def __init__(self):
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return {"observation": "start"}, {}

    def step(self, action):
        self.step_index += 1
        return (
            {"observation": f"step {self.step_index}"},
            1.0,
            False,
            False,
            {},
        )


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.log_prob = torch.nn.Parameter(torch.tensor(0.0))
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.value_observations = []

    def get_action(self, observation):
        return "action", float(self.log_prob.detach()), float(self.value.detach())

    def get_value(self, observation):
        self.value_observations.append(observation)
        return float(self.value.detach())

    def evaluate(self, observations, actions):
        count = len(observations)
        return (
            self.log_prob.expand(count),
            self.value.expand(count),
            self.log_prob * 0,
        )


def test_single_sample_update_is_finite_and_changes_policy():
    policy = TinyPolicy()
    trainer = PPOTrainer(TinyEnv(), policy, epochs=1, batch_size=8)
    trajectories = trainer.collect_trajectories(1)
    before = policy.value.detach().clone()

    metrics = trainer.train_step(
        trajectories,
        trainer.compute_advantages(trajectories),
    )

    assert torch.isfinite(torch.tensor(metrics["loss"]))
    assert not torch.equal(policy.value.detach(), before)


def test_train_bootstraps_from_the_last_next_observation():
    policy = TinyPolicy()
    trainer = PPOTrainer(TinyEnv(), policy, epochs=1, batch_size=2)

    trainer.train(num_iterations=1, steps_per_iteration=2)

    assert policy.value_observations[-1] == {"observation": "step 2"}


class Tokens(dict):
    def __init__(self, input_ids):
        super().__init__(input_ids=input_ids)
        self.input_ids = input_ids

    def to(self, device):
        return Tokens(self.input_ids.to(device))


class TinyTokenizer:
    def __call__(self, *_args, **_kwargs):
        return Tokens(torch.tensor([[2]]))

    def encode(self, _text, add_special_tokens=False):
        return [1]


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward(self, input_ids, output_hidden_states=False):
        batch, length = input_ids.shape
        zero = self.weight * 0
        logits = torch.stack(
            [zero, self.weight, -self.weight, zero]
        ).expand(batch, length, 4)
        hidden = self.weight.expand(batch, length, 2)
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


def test_llm_policy_evaluate_keeps_language_model_gradients():
    policy = LLMPolicy.__new__(LLMPolicy)
    torch.nn.Module.__init__(policy)
    policy.llm = TinyLM()
    policy.tokenizer = TinyTokenizer()
    policy.value_head = torch.nn.Linear(2, 1)

    log_probs, values, _entropy = policy.evaluate(
        [{"observation": "prompt"}],
        ["action"],
    )
    (log_probs.sum() + values.sum()).backward()

    assert policy.llm.weight.grad is not None
    assert policy.llm.weight.grad.abs().item() > 0
