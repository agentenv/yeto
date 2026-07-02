import pytest
import torch

from decoupled_diloco import losses


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
