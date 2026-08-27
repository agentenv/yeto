import torch

_ATOL = 1e-7


def _grpo_adam_step(initial, features, advantages):
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam(
        (parameter,),
        lr=1e-2,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0,
    )
    old_logprobs = features @ initial
    new_logprobs = features @ parameter
    ratios = torch.exp(new_logprobs - old_logprobs)
    loss = -(ratios * advantages).mean()
    loss.backward()
    optimizer.step()
    return parameter.detach()


def test_dense_h1_matches_central_on_cloned_groups_and_mean_oracle_otherwise():
    initial = torch.tensor([0.2, -0.3, 0.1], dtype=torch.float32)
    features_a = torch.tensor(
        [[1.0, 0.5, -0.5], [-0.25, 1.0, 0.75]], dtype=torch.float32
    )
    advantages_a = torch.tensor([1.0, -1.0], dtype=torch.float32)

    island_a = _grpo_adam_step(initial, features_a, advantages_a)
    island_b_cloned = _grpo_adam_step(initial, features_a, advantages_a)
    central_cloned = _grpo_adam_step(
        initial,
        features_a.repeat((2, 1)),
        advantages_a.repeat(2),
    )
    dense_cloned = (island_a + island_b_cloned) / 2
    assert torch.allclose(dense_cloned, central_cloned, rtol=0, atol=_ATOL)

    features_b = torch.tensor(
        [[0.75, -1.0, 0.25], [0.5, 0.25, 1.0]], dtype=torch.float32
    )
    advantages_b = torch.tensor([-0.4, 0.4], dtype=torch.float32)
    island_b = _grpo_adam_step(initial, features_b, advantages_b)
    dense_asymmetric = (island_a + island_b) / 2
    explicit_mean_of_local_deltas = (
        initial + ((island_a - initial) + (island_b - initial)) / 2
    )
    assert torch.allclose(
        dense_asymmetric,
        explicit_mean_of_local_deltas,
        rtol=0,
        atol=_ATOL,
    )
