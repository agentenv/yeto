import pytest
import torch

from yeto import losses


def _manual_sft_reference(logits, labels, weights=None):
    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    target_logprobs = torch.log_softmax(shift_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    token_weights = mask.to(target_logprobs.dtype)
    if weights is not None:
        token_weights = token_weights * weights[:, 1:].to(target_logprobs.dtype)
    return -(target_logprobs * token_weights).sum(), (token_weights > 0).sum()


def test_cross_entropy_matches_nll():
    lp = torch.log(torch.tensor([0.5, 0.25]))
    w = torch.ones(2)
    assert torch.isclose(losses.cross_entropy(lp, w), -(lp.sum()))


def test_importance_sampling_unit_ratio_reduces_to_pg():
    lp = torch.tensor([-1.0, -2.0])
    adv = torch.tensor([1.0, -0.5])
    assert torch.isclose(losses.importance_sampling(lp, lp, adv), -adv.sum())


def test_ppo_clips_high_ratio():
    target = torch.tensor([0.0])
    sampling = torch.tensor([-1.0])  # ratio = e ~ 2.72 > 1.2
    adv = torch.tensor([1.0])
    # min(r*A, clip(r)*A) = 1.2 with positive advantage
    assert torch.isclose(losses.ppo(target, sampling, adv), torch.tensor(-1.2))


def test_cispo_detaches_coefficient():
    target = torch.tensor([-1.0], requires_grad=True)
    sampling = torch.tensor([-1.0])
    adv = torch.tensor([2.0])
    loss = losses.cispo(target, sampling, adv)
    loss.backward()
    # d/dlp of -(sg(1.0) * lp * 2.0) = -2.0
    assert torch.isclose(target.grad, torch.tensor(-2.0))


def test_dro_penalizes_divergence():
    target = torch.tensor([-1.0])
    adv = torch.tensor([0.0])
    on_policy = losses.dro(target, torch.tensor([-1.0]), adv)
    off_policy = losses.dro(target, torch.tensor([-3.0]), adv)
    assert off_policy > on_policy


def test_sft_loss_masks_and_counts():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10)
    labels = torch.tensor([[-100, 3, -100, 7]])
    loss, n = losses.sft_loss(logits, labels)
    assert n == 2  # positions 1 and 3 (after shift)
    assert loss > 0


def test_sft_loss_rejects_rl_losses():
    with pytest.raises(ValueError):
        losses.sft_loss(torch.randn(1, 2, 4), torch.tensor([[1, 2]]), "ppo")


def test_sft_loss_weighted_equals_label_masking():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    weights = (torch.rand(2, 5) > 0.5).float()
    masked = labels.masked_fill(weights == 0, -100)
    weighted_loss, weighted_n = losses.sft_loss(logits, labels, weights=weights)
    masked_loss, masked_n = losses.sft_loss(logits, masked)
    assert torch.isclose(weighted_loss, masked_loss)
    assert weighted_n == masked_n


def test_sft_loss_counts_positive_weights_after_shift():
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 10)
    labels = torch.tensor([[5, 3, 6, 7]])
    weights = torch.tensor([[1.0, 1.0, 0.0, 1.0]])  # position 0 shifts away
    loss, n = losses.sft_loss(logits, labels, weights=weights)
    assert n == 2  # shifted weights [1, 0, 1]
    assert loss > 0


def test_sft_loss_zero_weights_contribute_nothing():
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[1, 2, 3]])
    loss, n = losses.sft_loss(logits, labels, weights=torch.zeros(1, 3))
    assert n == 0
    assert loss == 0


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sft_cross_entropy_loss_and_gradient_match_manual_reference(weighted, dtype):
    torch.manual_seed(123)
    labels = torch.randint(0, 17, (3, 7))
    labels[0, 2] = -100
    labels[2, 5] = -100
    weights = None
    if weighted:
        # Include zero, one, and arbitrary fractional weights. Position zero
        # is intentionally nonzero to exercise causal shifting.
        weights = torch.tensor(
            [
                [0.25, 1.0, 0.5, 0.0, 0.75, 1.0, 0.125],
                [1.0, 0.4, 0.0, 0.6, 1.0, 0.2, 0.8],
                [0.5, 1.0, 0.3, 0.7, 0.0, 0.9, 1.0],
            ]
        )

    native_logits = torch.randn(3, 7, 17, dtype=dtype, requires_grad=True)
    reference_logits = native_logits.detach().clone().requires_grad_(True)
    native_loss, native_tokens = losses.sft_loss(
        native_logits, labels, weights=weights
    )
    reference_loss, reference_tokens = _manual_sft_reference(
        reference_logits, labels, weights
    )
    native_loss.backward()
    reference_loss.backward()

    torch.testing.assert_close(native_loss, reference_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(
        native_logits.grad, reference_logits.grad, rtol=1e-6, atol=1e-6
    )
    assert native_tokens == reference_tokens


def test_custom_loss_from_file(tmp_path):
    f = tmp_path / "my_loss.py"
    f.write_text(
        "def loss_fn(logits, input_ids, weights):\n"
        "    return logits.sum() * 0, int((weights > 0).sum())\n"
    )
    fn = losses.load_custom_loss(f"custom:{f}")
    weights = torch.tensor([[1.0, 1.0, 0.0, 1.0]])
    loss, n = fn(torch.randn(1, 4, 8), torch.zeros(1, 4, dtype=torch.long), weights)
    assert loss == 0 and n == 3


def test_pickled_loss_roundtrip_with_closure(tmp_path):
    scale = 2.5  # captured by value — plain pickle could not ship this

    def weighted(logits, input_ids, weights):
        return logits.float().pow(2).sum() * scale, int(weights.sum())

    path = tmp_path / "loss.pkl"
    losses.dump_pickled_loss(weighted, path)
    fn = losses.load_pickled_loss(f"pickle:{path}")
    logits = torch.ones(1, 2, 3)
    loss, n = fn(logits, torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2))
    assert torch.isclose(loss, torch.tensor(6 * 2.5))
    assert n == 2
