from __future__ import annotations

import pytest
import torch

from yeto.megatron.hdo_gradient_stream import install_hdo_cpu_gradient_streaming


class _FakeHDO:
    def __init__(self, groups: list[dict], model_parameters: list[torch.Tensor]):
        self.offload_fraction = 1.0
        self.param_update_in_fp32 = True
        self.overlap_cpu_optimizer_d2h_h2d = False
        self.gpu_optimizer = None
        self.pin_cpu_grads = False
        self.cpu_copy_map_grad = {}
        self._cpu_optimizer_map_data_event = {}
        self.cpu_optimizers = [
            torch.optim.AdamW(
                groups,
                betas=(0.9, 0.95),
                eps=1e-8,
                fused=True,
            )
        ]
        cpu_parameters = [
            parameter
            for group in self.cpu_optimizers[0].param_groups
            for parameter in group["params"]
        ]
        self.cpu_copys_map_gpu_param = dict(
            zip(cpu_parameters, model_parameters, strict=True)
        )

        def copy_back(optimizer, args, kwargs):
            del args, kwargs
            for group in optimizer.param_groups:
                for cpu_parameter in group["params"]:
                    model_parameter = self.cpu_copys_map_gpu_param[cpu_parameter]
                    model_parameter.data.copy_(cpu_parameter.data)

        self.cpu_optimizers[0].register_step_post_hook(copy_back)

    def _set_sub_optimizer_grads(self):
        raise AssertionError("bulk gradient staging must be replaced")


def _parameters(optimizer: torch.optim.Optimizer) -> list[torch.Tensor]:
    return [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]


def _make_pair() -> tuple[_FakeHDO, torch.optim.AdamW, list[torch.Tensor]]:
    values = [
        torch.linspace(-1.0, 1.0, 3, dtype=torch.float32),
        torch.linspace(0.25, 1.25, 5, dtype=torch.float32),
        torch.linspace(-2.0, 0.5, 4, dtype=torch.float32),
    ]
    streamed_inner = [torch.nn.Parameter(value.clone()) for value in values]
    reference_inner = [torch.nn.Parameter(value.clone()) for value in values]
    models = [
        torch.nn.Parameter(value.to(torch.bfloat16), requires_grad=False)
        for value in values
    ]
    streamed_groups = [
        {
            "params": streamed_inner[:2],
            "lr": 2e-3,
            "weight_decay": 0.1,
        },
        {
            "params": streamed_inner[2:],
            "lr": 7e-4,
            "weight_decay": 0.0,
        },
    ]
    reference_groups = [
        {
            "params": reference_inner[:2],
            "lr": 2e-3,
            "weight_decay": 0.1,
        },
        {
            "params": reference_inner[2:],
            "lr": 7e-4,
            "weight_decay": 0.0,
        },
    ]
    hdo = _FakeHDO(streamed_groups, models)
    reference = torch.optim.AdamW(
        reference_groups,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=True,
    )
    return hdo, reference, models


def test_streamed_step_is_bit_exact_to_batched_adamw():
    hdo, reference, models = _make_pair()
    cpu_optimizer = hdo.cpu_optimizers[0]
    streamed_parameters = _parameters(cpu_optimizer)
    reference_parameters = _parameters(reference)
    optimizer_identity = id(cpu_optimizer)
    group_identities = [id(group) for group in cpu_optimizer.param_groups]

    scratch_bytes = install_hdo_cpu_gradient_streaming(
        hdo, require_cuda_gradients=False
    )
    full_gradient_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in streamed_parameters
    )
    assert scratch_bytes == max(
        parameter.numel() * parameter.element_size()
        for parameter in streamed_parameters
    )
    assert scratch_bytes < full_gradient_bytes
    assert hdo._yeto_bulk_gradient_bytes == full_gradient_bytes

    for step in range(4):
        gradients = []
        for index, parameter in enumerate(streamed_parameters):
            gradient = torch.linspace(
                -0.25 + step,
                0.75 - step / 4,
                parameter.numel(),
                dtype=torch.float32,
            )
            # Qwen's ordinary parameter gradients are BF16 while selected
            # precision-sensitive tensors (for example A_log) remain FP32.
            gradients.append(
                gradient
                if index == len(streamed_parameters) - 1
                else gradient.to(torch.bfloat16)
            )
        for model, reference_parameter, gradient in zip(
            models, reference_parameters, gradients, strict=True
        ):
            model.decoupled_grad = gradient
            reference_parameter.grad = None if gradient is None else gradient.float()

        hdo._set_sub_optimizer_grads()
        cpu_optimizer.step()
        reference.step()
        reference.zero_grad(set_to_none=True)

        assert id(cpu_optimizer) == optimizer_identity
        assert [id(group) for group in cpu_optimizer.param_groups] == group_identities
        for streamed, expected, model in zip(
            streamed_parameters, reference_parameters, models, strict=True
        ):
            assert torch.equal(streamed, expected)
            assert torch.equal(model, expected.to(torch.bfloat16))
            assert streamed.grad is None
            for key in ("step", "exp_avg", "exp_avg_sq"):
                assert torch.equal(
                    cpu_optimizer.state[streamed][key], reference.state[expected][key]
                )

    assert hdo.cpu_copy_map_grad == {}


def test_missing_source_gradient_fails_before_any_stale_update():
    hdo, _, models = _make_pair()
    cpu_optimizer = hdo.cpu_optimizers[0]
    parameters = _parameters(cpu_optimizer)
    install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)

    for model in models:
        model.decoupled_grad = torch.ones_like(model)
    hdo._set_sub_optimizer_grads()
    cpu_optimizer.step()

    parameter_before = [parameter.detach().clone() for parameter in parameters]
    state_before = [
        {
            key: value.detach().clone()
            for key, value in cpu_optimizer.state[parameter].items()
            if isinstance(value, torch.Tensor)
        }
        for parameter in parameters
    ]
    for model in models:
        model.decoupled_grad = torch.full_like(model, 2)
    # Stock dense HDO leaves the previous CPU grad attached here and silently
    # applies it twice. The bounded diagnostic must fail before touching any
    # master parameter, moment, or per-parameter step counter.
    models[-1].decoupled_grad = None

    with pytest.raises(RuntimeError, match="refusing stale-gradient reuse"):
        cpu_optimizer.step()

    for parameter, expected_parameter, expected_state in zip(
        parameters, parameter_before, state_before, strict=True
    ):
        assert torch.equal(parameter, expected_parameter)
        for key, expected in expected_state.items():
            assert torch.equal(cpu_optimizer.state[parameter][key], expected)
        assert parameter.grad is None


def test_streamed_step_rejects_closure_before_mutation():
    hdo, _, models = _make_pair()
    cpu_optimizer = hdo.cpu_optimizers[0]
    parameters = _parameters(cpu_optimizer)
    install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)
    models[0].decoupled_grad = torch.ones_like(models[0])
    before = [parameter.detach().clone() for parameter in parameters]

    with pytest.raises(RuntimeError, match="does not support closures"):
        cpu_optimizer.step(lambda: 1.0)

    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(parameters, before, strict=True)
    )
    assert not cpu_optimizer.state


def test_streamed_step_restores_groups_and_grads_after_adam_failure():
    hdo, _, models = _make_pair()
    cpu_optimizer = hdo.cpu_optimizers[0]
    parameters = _parameters(cpu_optimizer)
    install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)
    original_groups = list(cpu_optimizer.param_groups)
    original_group_params = [list(group["params"]) for group in original_groups]
    for model in models:
        model.decoupled_grad = torch.ones_like(model)

    def fail_step():
        raise RuntimeError("synthetic Adam failure")

    cpu_optimizer._yeto_original_step = fail_step
    with pytest.raises(RuntimeError, match="synthetic Adam failure"):
        cpu_optimizer.step()

    assert cpu_optimizer.param_groups == original_groups
    assert all(
        group["params"] == expected
        for group, expected in zip(original_groups, original_group_params, strict=True)
    )
    assert all(parameter.grad is None for parameter in parameters)


def test_streamed_step_rejects_mismatched_source_shape_and_restores_groups():
    hdo, _, models = _make_pair()
    cpu_optimizer = hdo.cpu_optimizers[0]
    install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)
    original_groups = list(cpu_optimizer.param_groups)
    for model in models:
        model.decoupled_grad = torch.ones_like(model)
    models[-1].decoupled_grad = torch.ones(models[-1].numel() + 1)

    with pytest.raises(RuntimeError, match="gradient size mismatch"):
        cpu_optimizer.step()

    assert cpu_optimizer.param_groups == original_groups
    assert all(parameter.grad is None for parameter in _parameters(cpu_optimizer))


def test_production_stream_rejects_non_cuda_source_gradient():
    hdo, _, models = _make_pair()
    cpu_optimizer = hdo.cpu_optimizers[0]
    install_hdo_cpu_gradient_streaming(hdo)
    models[0].decoupled_grad = torch.ones_like(models[0])

    with pytest.raises(RuntimeError, match="requires CUDA source gradients"):
        cpu_optimizer.step()

    assert all(parameter.grad is None for parameter in _parameters(cpu_optimizer))


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda hdo: setattr(hdo, "offload_fraction", 0.5), "offload_fraction"),
        (
            lambda hdo: setattr(hdo, "overlap_cpu_optimizer_d2h_h2d", True),
            "non-overlapped",
        ),
        (
            lambda hdo: hdo.cpu_copy_map_grad.update(
                {next(iter(hdo.cpu_copys_map_gpu_param)): torch.empty(1)}
            ),
            "retained CPU gradient",
        ),
    ],
)
def test_install_rejects_incompatible_hdo(mutation, message):
    hdo, _, _ = _make_pair()
    mutation(hdo)
    with pytest.raises((RuntimeError, TypeError), match=message):
        install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)


def test_install_rejects_initialized_adam_state():
    hdo, _, _ = _make_pair()
    optimizer = hdo.cpu_optimizers[0]
    parameter = _parameters(optimizer)[0]
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    parameter.grad = None

    with pytest.raises(RuntimeError, match="before Adam state initialization"):
        install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)


def test_source_mapping_must_be_bijective():
    hdo, _, _ = _make_pair()
    hdo.cpu_copys_map_gpu_param.pop(next(iter(hdo.cpu_copys_map_gpu_param)))
    with pytest.raises(RuntimeError, match="not bijective"):
        install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)


def test_install_rejects_duplicate_cpu_parameter():
    hdo, _, _ = _make_pair()
    parameter = _parameters(hdo.cpu_optimizers[0])[0]
    hdo.cpu_optimizers[0].param_groups[0]["params"].append(parameter)
    with pytest.raises(RuntimeError, match="CPU parameter list contains duplicates"):
        install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)


def test_install_rejects_duplicate_source_parameter():
    hdo, _, _ = _make_pair()
    parameters = _parameters(hdo.cpu_optimizers[0])
    hdo.cpu_copys_map_gpu_param[parameters[1]] = hdo.cpu_copys_map_gpu_param[
        parameters[0]
    ]
    with pytest.raises(
        RuntimeError, match="source-parameter mapping contains duplicates"
    ):
        install_hdo_cpu_gradient_streaming(hdo, require_cuda_gradients=False)
