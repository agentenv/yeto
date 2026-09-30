# SGLang patch port (task 1.3)

## Base

- Base: `sgl-project/sglang` branch `sglang-miles` @ `571212b636baca45e10fa3b4da11a289123f3235`.
- Why this commit: `radixark/miles@9e4260d` does not pin an SGLang commit. `docker/Dockerfile` has `SGLANG_BRANCH=sglang-miles` and an empty `SGLANG_COMMIT`, so the image builds from the branch HEAD at build time. `571212b` was the branch HEAD when Miles 9e4260d was committed (2026-09-28T04:55Z). The next branch commit, `e5bba1f` (a LoRA buffer-sizing cherry-pick), landed about 5h later.
- The cu12 variant in `docker/build.py` tracks a different branch, `sglang-miles-v0.5.19-final`. It is not used here.
- Working tree: `/home/michael/work/sglang-next`. Committed and pushed to michaellchung/sglang yeto/ports (HEAD 9f29303).
- Patches: `/home/michael/work/sglang-patches/NN-<srcsha>.patch`. Apply them in order with `git apply` on the base. The original source patches are kept next to them as `src-<sha>.patch`.
- Tests: CPU venv `/home/michael/work/sglang-next-venv` (torch 2.14.0+cpu), run with `PYTHONPATH=python`. With the full series applied, all 34 ported/affected tests pass.
- None of the five source patches applied cleanly (`git apply --check` failed for all of them). Each one was re-ported by hand.

## Upstream LoRA sync format (Miles 9e4260d) — context for 95d4d69/8db3711/c2cb40a

The upstream stack no longer has `/load_lora_adapter_from_tensors`: SGLang at the base has no such endpoint, request type or engine method. `SGLangApiClient.load_lora_adapter_from_tensors` is still in Miles, but nothing calls it. The live path is:

1. `POST /register_lora_adapter {lora_name, config_dict, pinned}` (the name is `miles_lora`).
2. Miles streams adapter tensors named `{lora_name}:{hf_key}` through the ordinary `POST /update_weights_from_tensor`. Each request carries one `flattened_bucket` per dtype, `serialized_named_tensors` holds one base64 payload per TP rank, and `selector` is set. The tensors are applied at `end_weight_update`.

SGLang upstream already serializes per rank and deserializes by `tp_rank` on this path. It also allows dynamic LoRA when `dp_size==1 or enable_dp_attention`.

## Per patch

| # | Source | Ported patch | Upstream equivalent? | Tests | Notes |
|---|---|---|---|---|---|
| 1 | `b34df47` release LoRA refs on abort | `01-b34df47.patch` (test only) | **Yes.** `TokenizerManager._finalize_lora_lease` is called from `_handle_abort_req` and releases each lease exactly once. | pass (27/27 in `test_tokenizer_manager_rid_cleanup.py`) | Only the regression test `test_abort_releases_lora_reference` is ported, adapted to `_make_tokenizer_manager(self)`. No production code changes. |
| 2 | `95d4d69` align tensor LoRA requests with Miles | `02-95d4d69.patch` (test only) | **Yes, functionally.** Upstream moved LoRA bytes onto `update_weights_from_tensor`, which already uses per-rank `serialized_named_tensors` + `normalize_serialized_named_tensor_payloads` + `[tp_rank]` deserialize. It also already has the dp-attention assertions. The `self.tp_rank` log bug is gone because that code was removed. | pass (2/2) | **Realigned:** the original code targeted the removed `LoadLoRAAdapterFromTensorsReqInput`. The new test `test/registered/unit/managers/test_miles_lora_wire_format.py` checks what Miles actually sends. (a) The `/register_lora_adapter` payload validates. (b) A 2-rank Miles-shaped `flattened_bucket` `/update_weights_from_tensor` payload (base64 str, `selector`) validates. Each rank then deserializes only its own copy, into `miles_lora:`-prefixed tensors. |
| 3 | `8db3711` multi-bucket tensor LoRA payloads | `03-8db3711.patch` | **Partly N/A.** The `flattened_buckets` load format belonged to the removed endpoint; Miles now sends one `flattened_bucket` request per dtype. **No upstream equivalent** for the second part: validation errors still echo the full request input. | pass; the new test fails without the fix and passes with it | **Realigned:** dropped `_reconstruct_flattened_tensor_buckets`. Kept the no-payload-reflection guard in `validation_exception_handler`, retargeted from `/load_lora_adapter_from_tensors` to `/update_weights_from_tensor` (`_TENSOR_PAYLOAD_PATHS`). Test: `test/registered/unit/entrypoints/test_http_server_validation.py`. |
| 4 | `c2cb40a` preserve multi-bucket LoRA engine payloads | none (not ported) | **N/A.** It only changed `Engine.load_lora_adapter_from_tensors` (removed upstream) to accept `flattened_buckets`, plus a GPU e2e test on that API. | skip-GPU (the original test is a GPU e2e test on a removed API) | Nothing in the upstream stack needs this. |
| 5 | `e1b57eb` disk-backed weight memory saver | `05-e1b57eb.patch` | **No.** | pass (4/4); all 4 fail without the fix. Server-args ratchets/namespace/CLI-metadata tests: 13 pass | Ported onto the new config layout. The field `enable_weights_disk_backup` goes in `arg_groups/fields/exec_.py` and `field_order.py` (not `ServerArgs`). It is read through `get_exec().features`, only for the main model (never the draft model), and must not be combined with CPU backup (assert). `region(..., enable_disk_backup=)` is passed through by the Real/Noop adapters. Two tests were added: one checks the draft model never uses disk backup, the other checks the mutual-exclusion assert. The real disk spill needs a GPU and is skipped. `torch_memory_saver@b5588e8` (Miles Dockerfile) and PyPI 0.0.10 both accept `enable_disk_backup`. |

## Risks / follow-ups

- The base is inferred from the branch HEAD at the time of the Miles commit, because Miles does not pin SGLang. Before task 1.5 sets `SGLANG_NEXT_COMMIT`, confirm that the `lmsysorg/sglang:v0.5.20`-based image intended for ports can run `571212b`. Also decide whether to take `e5bba1f` (LoRA buffer sizing), which is newer.
- Patches 1 and 2 are test-only regression guards. Only patches 3 and 5 change runtime behavior.
- Nothing was run on a GPU. That means no real LoRA publish, no CUDA-IPC round trip, and no TMS disk spill. The task 1.3 statement "由真实 LoRA 发布测试兜底" (backstopped by a real LoRA publish test) still needs a GPU run.
- Committed and pushed to michaellchung/sglang yeto/ports (HEAD 9f29303). Creating `michaellchung/sglang` `yeto/ports` and making one commit per patch still needs authorization.
- One upstream test was already failing before these patches and is unrelated to them: `test_msgpack_ipc_roundtrip.py::test_check_weights_mirrors_match_pydantic_models` (field `role`).

## 用户确认（2026-09-29）

用户确认：`b34df47`、`95d4d69` 因 upstream 已有等价实现只移植测试，`c2cb40a` 因只涉及 upstream 已删除的接口而不移植，这三项处理均认可。移植后的提交已推送到 `michaellchung/sglang` 的 `yeto/ports`（HEAD `9f29303`），`SGLANG_NEXT_COMMIT` 已固定在该提交。
