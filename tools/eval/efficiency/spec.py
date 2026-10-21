"""One shared spec per model preset, and the command plan of every arm.

An arm is ``<system>-<framework>-<mode>``:

* system ``bare`` (the framework's own entry point, no yeto code on its path)
  or ``yeto`` (the same framework driven by yeto);
* framework ``verl`` (fork pin ``VERL_COMMIT``, the verl image yeto builds) or
  ``miles`` (``MILES_NEXT_IMAGE``, the image yeto runs Miles in);
* mode ``sync`` (one card, rollout and trainer colocated) or ``async``
  (two cards, one rollout + one trainer, fully_async with a policy-age limit).

Every value the comparison depends on comes from :class:`EffSpec`;
:func:`control_fingerprint` hashes them and every arm records it.

Sources of the bare command lines (the "official recipe" each one follows):

* bare verl sync: ``examples/grpo_trainer/run_qwen3_4b_fsdp.sh`` (fork
  acad9875), ``python -m verl.trainer.main_ppo`` with its override groups;
* bare verl async: ``verl/experimental/fully_async_policy/shell/
  dapo_7b_math_fsdp2_4_4.sh``, ``python -m verl.experimental.fully_async_policy.fully_async_main``;
* bare Miles sync: ``scripts/run_qwen3_4b.py`` (Megatron branch, bridge mode)
  -> ``train.py``; model flags from the image's own ``scripts/models/<file>``;
* bare Miles async: ``examples/infra_features/fully_async/`` (``--fully-async``)
  -> ``train_async.py``.

Values the spec does not name stay at the recipe's value, or at the
framework default where the recipe sets none. Where yeto sets a value the
recipe also sets (LoRA, TIS, dtype), the bare arm takes the spec value, so the
two arms differ only by what yeto adds; :func:`verl_knob_diff` lists the verl
overrides that still differ.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

SYSTEMS = ("bare", "yeto")
FRAMEWORKS = ("verl", "miles")
MODES = ("sync", "async")
ARMS = tuple(f"{s}-{f}-{m}" for f in FRAMEWORKS for m in MODES for s in SYSTEMS)

EFF_DIR = "/root/eff"  # spec, reward and prep script inside a bare container
OUT_DIR = "/root/eff-out"  # everything a container writes for the results Volume
DATA_DIR = "/root/eff-data"  # eff_prep.py output
MODEL_DIR = "/root/eff-models"
VERL_ROOT = "/workspace/verl"
MILES_ROOT = "/root/miles"
MEGATRON_ROOT = "/root/Megatron-LM"  # yeto.launch.remote_setup.PORTS_MEGATRON_PATH


@dataclass(frozen=True)
class EffSpec:
    """Everything the arms of one comparison share."""

    name: str
    model: str
    model_revision: str
    miles_model_args: str  # file under <MILES_ROOT>/scripts/models/ (the official model flags)
    data: str = "zhuzilin/gsm8k"
    data_revision: str = "0cbd9f31d91ac21a7613dcbc7fef992adac459ae"
    data_file: str = ""  # one JSONL file of the dataset repo (prompt, label); "" = train/test.parquet
    train_rows: int = 0  # keep the first rows only (0 = all); needed with data_file
    gpu: str = "A10G"  # Modal GPU type, the same for every arm
    rounds: int = 8  # optimizer steps (= policy versions) per run
    groups_per_round: int = 8  # prompts per step
    samples_per_group: int = 4  # responses per prompt
    max_prompt_len: int = 512
    max_response_len: int = 256  # "max_tokens" of the rollout
    lr: float = 1e-5
    lora_r: int = 32  # yeto's verl backend is LoRA only, so every arm is LoRA
    seed: int = 17
    temperature: float = 1.0
    top_p: float = 1.0
    mem_fraction: float = 0.5  # rollout engine static memory fraction
    max_policy_age: int = 1  # async arms: versions a sample may lag
    tis_upper: float = 2.0  # token-level truncated IS upper bound (all arms)
    repeats: int = 2  # runs per arm (range verdicts need >= 2)
    timeout_s: int = 3000
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    # yeto arms only (the bare arms keep no checkpoint): "final" cuts once at the end
    # (fair control), "every-round" is the product default (a round cut every round).
    # Not in the control fingerprint: it does not change what the bare arms run.
    checkpoint: str = "final"
    # cards per run: 0 keeps the E0 shape (sync 1, async 1 rollout + 1 trainer); E11
    # (whole-cluster comparison) sets the total, split evenly between rollout and trainer in async
    cards: int = 0
    # "lora" or "full" (every parameter; E11 0.6B). Miles full-parameter islands with
    # engines need DP=1, so a 2-card trainer runs tensor parallel 2 in every arm.
    tuning: str = "lora"
    tensor_parallel: int = 1

    @property
    def samples_per_round(self) -> int:
        return self.groups_per_round * self.samples_per_group

    @property
    def seq_len(self) -> int:
        return self.max_prompt_len + self.max_response_len

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRESETS: dict[str, EffSpec] = {
    # E0.1 acceptance (debug size): the T-EVAL model and data on the T-EVAL card
    "qwen3-0.6b": EffSpec(
        name="qwen3-0.6b", model="Qwen/Qwen3-0.6B", model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        miles_model_args="qwen3-0.6B.py", chat_template_kwargs={"enable_thinking": False}),
    # E0.2 (main comparison): revision read from the HF API 2026-10-20
    "qwen3-4b-instruct-2507": EffSpec(
        name="qwen3-4b-instruct-2507", model="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="cdbee75f17c01a7cc42f958dc650907174af0554", miles_model_args="qwen3-4B-Instruct-2507.py",
        gpu="H100!", rounds=20, groups_per_round=32, samples_per_group=8, max_prompt_len=512,
        max_response_len=1024, lora_r=64, mem_fraction=0.6, timeout_s=5400),
    # E11 (whole cluster): the E0.2 model and hyperparameters on 4 cards; async (2 rollout +
    # 2 trainer, or 2 islands of 1 + 1), so the 2-island yeto arm and the single-island arms
    # share one shape. Data: DAPO-Math-17k (gsm8k is saturated for this model: reward 0.97
    # in round 1 of e02-bare-verl-sync-1), the first rounds x groups rows (one epoch).
    "e11-qwen3-4b": EffSpec(
        name="e11-qwen3-4b", model="Qwen/Qwen3-4B-Instruct-2507",
        model_revision="cdbee75f17c01a7cc42f958dc650907174af0554", miles_model_args="qwen3-4B-Instruct-2507.py",
        data="zhuzilin/dapo-math-17k", data_revision="2e65612930298bde4c5d58fd97b3f23a483aaff9",
        data_file="dapo-math-17k.jsonl", train_rows=12 * 32,
        gpu="H100!", rounds=12, groups_per_round=32, samples_per_group=8, max_prompt_len=512,
        max_response_len=1024, lora_r=64, mem_fraction=0.6, timeout_s=5400, cards=4),
    # E11 full parameter: the E11 data, rounds and lengths on Qwen3-0.6B, every parameter
    # trained (lora_r is unused). 2 rollout + 2 trainer cards, trainer TP 2 (DP 1).
    "e11-qwen3-0.6b-full": EffSpec(
        name="e11-qwen3-0.6b-full", model="Qwen/Qwen3-0.6B", model_revision="c1899de289a04d12100db370d81485cdf75e47ca",
        miles_model_args="qwen3-0.6B.py",
        data="zhuzilin/dapo-math-17k", data_revision="2e65612930298bde4c5d58fd97b3f23a483aaff9",
        data_file="dapo-math-17k.jsonl", train_rows=12 * 32,
        gpu="H100!", rounds=12, groups_per_round=32, samples_per_group=8, max_prompt_len=512,
        max_response_len=1024, mem_fraction=0.6, timeout_s=3600, cards=4, tuning="full",
        tensor_parallel=2, chat_template_kwargs={"enable_thinking": False}),
}


@dataclass(frozen=True)
class Arm:
    system: str
    framework: str
    mode: str

    @classmethod
    def parse(cls, name: str) -> "Arm":
        parts = name.split("-")
        if len(parts) != 3 or name not in ARMS:
            raise ValueError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
        return cls(*parts)

    @property
    def name(self) -> str:
        return f"{self.system}-{self.framework}-{self.mode}"

    @property
    def peer(self) -> str:
        """The arm this one is compared with (same framework and mode)."""
        return f"{'yeto' if self.system == 'bare' else 'bare'}-{self.framework}-{self.mode}"


def get_spec(preset: str, **overrides: Any) -> EffSpec:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {', '.join(PRESETS)}")
    return replace(PRESETS[preset], **overrides) if overrides else PRESETS[preset]


def cards(mode: str, total: int = 0) -> dict[str, int]:
    """Cards per role: sync colocates both on every card; async splits them evenly
    (``total`` 0: one card for sync, one of each role for async)."""
    if mode == "sync":
        n = total or 1
        return {"total": n, "rollout": n, "trainer": n, "colocated": True}
    n = total or 2
    if n % 2:
        raise ValueError(f"async needs an even card count (rollout + trainer halves), got {n}")
    return {"total": n, "rollout": n // 2, "trainer": n // 2, "colocated": False}


def control_fingerprint(spec: EffSpec) -> str:
    """sha256 over every shared value except the repeat count (a report refuses
    to compare runs whose fingerprints differ)."""
    doc = {k: v for k, v in spec.to_dict().items() if k not in ("repeats", "timeout_s", "checkpoint")}
    # fields added after E0.1: a default value keeps the fingerprints of earlier runs valid
    for key, default in (("cards", 0), ("data_file", ""), ("train_rows", 0), ("tuning", "lora"),
                         ("tensor_parallel", 1)):
        if doc[key] == default:
            doc.pop(key)
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()


def images() -> dict[str, str]:
    """The image pins every arm of a framework uses (same image for bare and yeto)."""
    from yeto.rl.adapters.miles.pins import MILES_NEXT_IMAGE
    from yeto.rl.adapters.verl.pins import VERL_COMMIT

    return {"verl": f"verl-build:{VERL_COMMIT}", "miles": MILES_NEXT_IMAGE}


def _model_path(spec: EffSpec) -> str:
    return f"{MODEL_DIR}/{spec.model.split('/')[-1]}"


# ====================================================================== bare verl
def verl_bare_overrides(spec: EffSpec, mode: str, run_id: str) -> list[str]:
    """Hydra overrides of the bare verl arm (recipe groups in the recipe's order)."""
    c = cards(mode, spec.cards)
    data = [
        "algorithm.adv_estimator=grpo",
        "algorithm.use_kl_in_reward=False",
        "algorithm.norm_adv_by_std_in_grpo=True",
        f"data.train_files={DATA_DIR}/train.parquet",
        f"data.val_files={DATA_DIR}/test.parquet",
        f"data.max_prompt_length={spec.max_prompt_len}",
        f"data.max_response_length={spec.max_response_len}",
        "data.filter_overlong_prompts=True",
        "data.truncation=error",
        f"data.seed={spec.seed}",
        *[f"+data.apply_chat_template_kwargs.{k}={v}" for k, v in sorted(spec.chat_template_kwargs.items())],
    ]
    model = [
        f"actor_rollout_ref.model.path={_model_path(spec)}",
        "actor_rollout_ref.model.use_remove_padding=True",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        f"actor_rollout_ref.model.lora_rank={spec.lora_r}",
        f"actor_rollout_ref.model.lora_alpha={spec.lora_r}",
        "actor_rollout_ref.model.target_modules=all-linear",
    ]
    actor = [
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype=fp32",  # yeto's VerlRunConfig default
        f"actor_rollout_ref.actor.optim.lr={spec.lr!r}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={spec.groups_per_round}",
        "actor_rollout_ref.actor.use_dynamic_bsz=True",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.entropy_coeff=0",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False",
        f"actor_rollout_ref.actor.data_loader_seed={spec.seed}",
    ]
    rollout = [
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={spec.mem_fraction}",
        f"actor_rollout_ref.rollout.n={spec.samples_per_group}",
        f"actor_rollout_ref.rollout.temperature={spec.temperature}",
        f"actor_rollout_ref.rollout.top_p={spec.top_p}",
        "actor_rollout_ref.rollout.top_k=-1",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True",
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384",
        "algorithm.rollout_correction.rollout_is=token",
        f"algorithm.rollout_correction.rollout_is_threshold={spec.tis_upper}",
        "algorithm.rollout_correction.bypass_mode=False",
    ]
    reward = [f"reward.custom_reward_function.path={EFF_DIR}/eff_reward.py",
              "reward.custom_reward_function.name=compute_score"]
    trainer = [
        "trainer.critic_warmup=0",
        "trainer.logger=[console,file]",
        "trainer.project_name=yeto-effeval",
        f"trainer.experiment_name={run_id}",
        f"trainer.default_local_dir={OUT_DIR}/ckpt",
        "trainer.nnodes=1",
        "trainer.val_before_train=False",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.resume_mode=disable",
        f"trainer.total_training_steps={spec.rounds}",
        "trainer.total_epochs=1000",
        "ray_kwargs.ray_init.runtime_env.py_executable=null",
        "ray_kwargs.ray_init.num_cpus=32",  # a Modal container reports too few Ray CPUs (S17 V2)
    ]
    if mode == "sync":
        return [*data, f"data.train_batch_size={spec.groups_per_round}", *model, *actor, *rollout, *reward,
                *trainer, f"trainer.n_gpus_per_node={c['total']}"]
    fully_async = [  # dapo_7b_math_fsdp2_4_4.sh; one parameter version per spec round
        "data.train_batch_size=0",
        "data.gen_batch_size=1",
        "data.return_raw_chat=True",
        "actor_rollout_ref.hybrid_engine=False",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.checkpoint_engine.backend=nccl",
        # The fork's disaggregated sender drops the PEFT config of unmerged LoRA params
        # (engine_workers.update_weights), so the rollout would load them as base
        # weights; merged LoRA sends full weights, the fork's working LoRA path.
        "actor_rollout_ref.model.lora.merge=True",
        "rollout.nnodes=1",
        f"rollout.n_gpus_per_node={c['rollout']}",
        f"rollout.total_rollout_steps={spec.groups_per_round * spec.rounds}",
        f"trainer.n_gpus_per_node={c['trainer']}",
        f"async_training.staleness_threshold={spec.max_policy_age}",
        "async_training.trigger_parameter_sync_step=1",
        "async_training.require_batches=1",
        "async_training.partial_rollout=True",
        "async_training.use_trainer_do_validate=False",
        "async_training.use_dynamic_resource_scheduling=False",
    ]
    return [*data, *model, *actor, *rollout, *reward, *trainer, *fully_async]


def verl_bare_entry(mode: str) -> str:
    return "verl.trainer.main_ppo" if mode == "sync" else "verl.experimental.fully_async_policy.fully_async_main"


# ====================================================================== bare Miles
def miles_bare_argv(spec: EffSpec, mode: str) -> list[str]:
    """Miles arguments after the model flags (the container prepends ``$MODEL_ARGS``)."""
    c = cards(mode, spec.cards)
    model = _model_path(spec)
    argv = [
        # checkpoint: bridge mode loads the HF checkpoint directly (run_qwen3_4b.py --megatron-to-hf-mode bridge)
        "--hf-checkpoint", model, "--ref-load", model, "--megatron-to-hf-mode", "bridge",
        "--finetune", "--no-load-optim", "--no-load-rng", "--no-save-optim", "--no-save-rng",
        # rollout
        "--prompt-data", f"{DATA_DIR}/train.jsonl", "--input-key", "prompt", "--label-key", "label",
        # no --rollout-shuffle (the recipe has it): yeto's Miles argv leaves rollout_shuffle
        # False (eff01-yeto-miles-* Miles arg dump), and both arms must read prompts in one order
        "--apply-chat-template", "--rollout-seed", str(spec.seed),
        "--custom-rm-path", "eff_reward.score",
        "--num-rollout", str(spec.rounds), "--rollout-batch-size", str(spec.groups_per_round),
        "--n-samples-per-prompt", str(spec.samples_per_group),
        "--rollout-max-response-len", str(spec.max_response_len),
        "--rollout-max-context-len", str(spec.seq_len),
        "--rollout-temperature", str(spec.temperature), "--rollout-top-p", str(spec.top_p),
        "--global-batch-size", str(spec.samples_per_round), "--balance-data",
        # GRPO + TIS (the correction every arm uses)
        "--advantage-estimator", "grpo", "--entropy-coef", "0.00", "--eps-clip", "0.2", "--eps-clip-high", "0.28",
        "--use-tis", "--tis-clip", str(spec.tis_upper),
        # optimizer
        "--optimizer", "adam", "--lr", str(spec.lr), "--lr-decay-style", "constant",
        "--seed", str(spec.seed),
        *_miles_lora_flags(spec),
        # Megatron (run_qwen3_4b.py, Megatron branch)
        "--train-backend", "megatron", "--seq-length", str(spec.seq_len),
        "--tensor-model-parallel-size", str(spec.tensor_parallel), "--pipeline-model-parallel-size", "1",
        "--context-parallel-size", "1", "--expert-model-parallel-size", "1", "--expert-tensor-parallel-size", "1",
        "--recompute-granularity", "full", "--recompute-method", "uniform", "--recompute-num-layers", "1",
        "--use-dynamic-batch-size", "--max-tokens-per-gpu", str(max_tokens_per_gpu(spec)),
        "--attention-dropout", "0.0", "--hidden-dropout", "0.0",
        "--accumulate-allreduce-grads-in-fp32", "--attention-softmax-in-fp32",
        # yeto's generic recipe (Qwen3) trains with "unfused": TE's flash path picks the
        # FlashAttention cute kernel, which fails to compile on A10G (sm86, eff01-bare-miles-sync-a).
        "--attention-backend", "unfused",
        "--bf16",
        # SGLang
        "--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", str(spec.mem_fraction),
        # placement
        "--actor-num-nodes", "1", "--actor-num-gpus-per-node", str(c["trainer"]),
        "--num-gpus-per-node", str(c["total"]),
    ]
    if spec.chat_template_kwargs:
        argv += ["--apply-chat-template-kwargs", json.dumps(spec.chat_template_kwargs, separators=(",", ":"))]
    if mode == "sync":
        argv += ["--colocate", "--offload-train"]
        if spec.tuning == "lora":
            argv += ["--lora-base-cpu-backup"]
    else:
        argv += ["--rollout-num-gpus", str(c["rollout"]), "--fully-async",
                 "--max-weight-staleness", str(spec.max_policy_age)]
    return argv


def _miles_lora_flags(spec: EffSpec) -> list[str]:
    """Miles' own LoRA flags (the same set yeto's Miles adapter emits); none for full parameter."""
    if spec.tuning == "full":
        return []
    return ["--lora-rank", str(spec.lora_r), "--lora-alpha", str(spec.lora_r), "--lora-dropout", "0",
            "--lora-type", "canonical_lora", "--target-modules", "all-linear",
            "--sglang-max-lora-rank", str(spec.lora_r)]


def _yeto_tuning(spec: EffSpec) -> str:
    """The yeto launcher's tuning options (full parameter: a no-sync single island)."""
    if spec.tuning == "full":
        return f"--tuning full --tensor-parallel {spec.tensor_parallel} "
    return f"--tuning lora --lora-r {spec.lora_r} "


def miles_bare_script(mode: str) -> str:
    return "train.py" if mode == "sync" else "train_async.py"


def max_tokens_per_gpu(spec: EffSpec) -> int:
    """Miles dynamic-batching budget (run_qwen3_4b.py: 9216), never below one sample."""
    return max(9216, spec.seq_len)


# ====================================================================== yeto arms
YETO_WORKDIR = "/root/sky_workdir"  # modal_verl.WORKDIR: the repo inside the yeto+verl container
ALGORITHM_SPEC = "examples/rl_algorithms/decoupled-ppo-c2.json"  # token TIS 2.0, max_policy_staleness 2
# the same with the bare Miles recipe's clip range (--eps-clip 0.2 --eps-clip-high 0.28)
MILES_ALGORITHM_SPEC = "tools/eval/efficiency/algo-eff-miles.json"
def yeto_verl_argv(spec: EffSpec, mode: str, tape: str) -> list[str]:
    """``island_entry`` arguments of the yeto+verl arm (the T-EVAL E1 shape)."""
    if uses_task_file(spec):
        raise ValueError("the yeto+verl arm reads --data from the Hub; it has no task-file path")
    argv = ["-m", "yeto.rl.adapters.verl.island_entry", "--model", spec.model, "--model-revision",
            spec.model_revision, "--data", spec.data, "--data-revision", spec.data_revision, "--learner-id", "0",
            "--reward-function", "gsm8k_reward:score", "--global-rounds", str(spec.rounds),
            "--groups-per-round", str(spec.groups_per_round), "--samples-per-group", str(spec.samples_per_group),
            "--rollout-max-response-len", str(spec.max_response_len), "--event-tape", tape,
            "--inner-lr", str(spec.lr), "--lora-r", str(spec.lora_r), "--lora-targets", "all-linear",
            "--seq-len", str(spec.seq_len), "--seed", str(spec.seed), "--rl-single-island-no-sync"]
    if spec.chat_template_kwargs:
        argv += ["--apply-chat-template-kwargs", json.dumps(spec.chat_template_kwargs)]
    if mode == "async":
        # The default algorithm spec allows no staleness (island_entry rejects age 1,
        # eff01-yeto-verl-async-a); use the yeto+Miles arm's spec: token TIS 2.0, staleness 2.
        argv += ["--rl-max-policy-age", str(spec.max_policy_age),
                 "--rl-algorithm-spec", f"{YETO_WORKDIR}/{ALGORITHM_SPEC}"]
    return argv


CHECKPOINT_MODES = ("final", "every-round")


def cut_every(spec: EffSpec) -> int:
    """Round-cut interval of the yeto+Miles arm: ``final`` cuts only at the end (the
    bare arms keep no checkpoint), ``every-round`` is the product default.
    The yeto+verl arm sets no checkpoint store and fully_async refuses cuts, so it
    has no round cut in either mode."""
    if spec.checkpoint not in CHECKPOINT_MODES:
        raise ValueError(f"checkpoint must be one of {CHECKPOINT_MODES}, got {spec.checkpoint!r}")
    return spec.rounds if spec.checkpoint == "final" else 1


# E1: the GPU model every arm must run on, per spec.gpu (as tools/eval/modal_verl.GPU_NAMES:
# Modal "A10G" containers are an NVIDIA A10 or an A10G, and the A10G trains slower)
GPU_MODELS = {"A10G": "NVIDIA A10", "L4": "L4", "H100!": "H100"}


def gpu_name_arg(spec: EffSpec) -> str:
    """The launcher option that pins the yeto+Miles arm to the bare arms' GPU model."""
    return f"--modal-gpu-name {shlex.quote(GPU_MODELS[spec.gpu])} "


def yeto_miles_launcher_args(spec: EffSpec, mode: str, run_id: str) -> str:
    """The yeto launcher arguments of the yeto+Miles arm (``tools/eval/miles_local.sh`` ARGS)."""
    from ..miles import STORE_VOLUME, TAPE_VOLUME

    card = {"A10G": "a10g", "H100!": "h100", "L4": "l4"}[spec.gpu]
    c = cards(mode, spec.cards)
    kwargs = json.dumps(spec.chat_template_kwargs) if spec.chat_template_kwargs else ""
    args = (f"launch --training-mode rl --rl-engine ports --preflight-threads off --modal-gpu-exact "
            f"{gpu_name_arg(spec)}"
            # container resources as the bare arms' (modal_eff: 8 CPU + 48 GiB per card)
            f"--modal-retries 0 --modal-cpu {8 * c['total']} --modal-memory {48 * c['total']} "
            f"--modal-tape-volume {TAPE_VOLUME} "
            f"--controller local --cluster-prefix {run_id} {_yeto_tuning(spec)}"
            f"--rl-observe-timeline --model {spec.model} --model-revision {spec.model_revision} {_yeto_data(spec)} "
            f"--rl-checkpoint-store modal-volume://{STORE_VOLUME}/{run_id} "
            f"--rl-cut-every {cut_every(spec)} "
            "--reward-function gsm8k_reward:score --fragments 1 --pipeline 1 "
            f"--rollout-batch-size {spec.groups_per_round} --n-samples-per-prompt {spec.samples_per_group} "
            f"--rollout-max-response-len {spec.max_response_len} --seq-len {spec.seq_len} "
            f"--inner-lr {spec.lr} --seed {spec.seed} --sglang-mem-fraction-static {spec.mem_fraction} "
            f"--gpu modal:{c['total']}x{card} --rl-single-island-no-sync --total-steps {spec.rounds} "
            # Miles recipe settings of the bare arm (S26 GOAL §6: the bare framework's official
            # settings win): no deterministic inference, constant LR, full recompute, dynamic
            # batching, gradient-accumulation fusion; dropout 0 is the generic recipe's own
            "--no-sglang-deterministic-inference --rl-lr-schedule constant --rl-activation-recompute "
            f"--rl-max-tokens-per-gpu {max_tokens_per_gpu(spec)} --rl-gradient-accumulation-fusion "
            # TIS (upper 2.0) and eps_clip_high 0.28 as in the bare arms; staleness 2 >= max_policy_age
            f"--trust-remote-code --rl-algorithm-spec {MILES_ALGORITHM_SPEC}")
    if kwargs:
        args += f" --apply-chat-template-kwargs {shlex.quote(kwargs)}"
    if mode == "async":
        args += (f" --rl-placement fixed-partition --rl-rollout-gpus {c['rollout']} "
                 f"--rl-max-policy-age {spec.max_policy_age}")
    return args


YETO_TASK_FILE = "eff-tasks.jsonl"  # the local task file in the launcher's checkout (local_task_file)


def uses_task_file(spec: EffSpec) -> bool:
    """The yeto+Miles arms read a local task file (eff_prep.yeto_task_rows) unless the
    spec is the full gsm8k train split, which they read from the Hub as before."""
    return bool(spec.data_file or spec.train_rows)


def write_task_file(spec: EffSpec, path: Path) -> Path:
    """Write the yeto arms' local task file (the bare arms' training rows, in order)."""
    from huggingface_hub import hf_hub_download

    from .eff_prep import yeto_task_rows

    path.write_text("".join(json.dumps(r) + "\n" for r in yeto_task_rows(spec.to_dict(), hf_hub_download)))
    return path


def _yeto_data(spec: EffSpec) -> str:
    from ..miles import DATA

    if uses_task_file(spec):
        return f"--rl-allow-local-data --data {YETO_TASK_FILE}"
    if spec.data != "zhuzilin/gsm8k" or spec.data_revision not in DATA:
        raise ValueError("the yeto+Miles arm reads tools.eval.miles.DATA; keep the spec's data equal to it")
    return DATA


HEAD_REGION = "nebius/eu-north1"  # the multi-island syncer head: a CPU-only Nebius VM


def yeto_miles_islands_launcher_args(spec: EffSpec, islands: int, run_id: str, *, data: str,
                                     island_tasks: str) -> str:
    """E11: the yeto+Miles arm split into ``islands`` async islands that sync through a
    head syncer (LoRA strict-avg), with the card total and the prompts per round of the
    single-island arm. ``data`` is a local JSONL of the spec's rows and ``island_tasks``
    the ``--rl-island-tasks`` spec that gives each island its share of every round."""
    c = cards("async", spec.cards)
    if c["total"] % islands or (c["total"] // islands) % 2 or spec.groups_per_round % islands:
        raise ValueError(f"{islands} islands need an even card share and an equal prompt share "
                         f"({c['total']} cards, {spec.groups_per_round} prompts per round)")
    per = c["total"] // islands
    card = {"A10G": "a10g", "H100!": "h100", "L4": "l4"}[spec.gpu]
    kwargs = json.dumps(spec.chat_template_kwargs) if spec.chat_template_kwargs else ""
    args = (f"launch --controller head --syncer-region {HEAD_REGION} --on-demand --training-mode rl "
            f"--rl-engine ports --preflight-threads wait --preflight-thread-start 2950 --modal-gpu-exact "
            f"{gpu_name_arg(spec)}--modal-retries 0 --no-island-relaunch "
            f"--modal-cpu {8 * per} --modal-memory-gib {48 * per} "
            f"--cluster-prefix {run_id} --tuning lora --lora-r {spec.lora_r} --lora-targets all-linear "
            f"--rl-observe-timeline --model {spec.model} --model-revision {spec.model_revision} "
            f"--rl-allow-local-data --data {data} --rl-island-tasks {shlex.quote(island_tasks)} "
            "--reward-function gsm8k_reward:score --fragments 1 --pipeline 1 "
            f"--rollout-batch-size {spec.groups_per_round // islands} "
            f"--n-samples-per-prompt {spec.samples_per_group} "
            f"--rollout-max-response-len {spec.max_response_len} --seq-len {spec.seq_len} "
            f"--inner-lr {spec.lr} --seed {spec.seed} --sglang-mem-fraction-static {spec.mem_fraction} "
            f"--gpu {','.join([f'modal:{per}x{card}'] * islands)} --total-steps {spec.rounds} "
            # the single-island arm's Miles recipe settings (yeto_miles_launcher_args)
            "--no-sglang-deterministic-inference --rl-lr-schedule constant --rl-activation-recompute "
            f"--rl-max-tokens-per-gpu {max_tokens_per_gpu(spec)} --rl-gradient-accumulation-fusion "
            f"--trust-remote-code --rl-algorithm-spec {MILES_ALGORITHM_SPEC} "
            # elastic: the multi-island LoRA path with real-card runs (S24 d4, S25 c3); legacy
            # scheduling hung in the first apply on both islands (e11-yeto-2i-1, no round in 13 min)
            "--rl-island-scheduling elastic "
            f"--rl-placement fixed-partition --rl-rollout-gpus {per // 2} "
            f"--rl-max-policy-age {spec.max_policy_age}")
    if kwargs:
        args += f" --apply-chat-template-kwargs {shlex.quote(kwargs)}"
    return args


def island_task_split(groups_per_round: int, rounds: int, islands: int) -> dict[str, list[str]]:
    """Round-major split of the first ``rounds * groups_per_round`` rows (task ids
    ``gsm<row+1>``): island k gets the k-th equal slice of every round's prompts, so the
    islands together read the single-island arm's prompts of each round."""
    share = groups_per_round // islands
    return {f"l{k}": [f"gsm{r * groups_per_round + k * share + i + 1}" for r in range(rounds)
                      for i in range(share)] for k in range(islands)}


# ====================================================================== plans
def bare_script(spec: EffSpec, arm: Arm, run_id: str) -> str:
    """The bash script a bare container runs (after eff_prep.py staged model and data)."""
    c = cards(arm.mode, spec.cards)
    lines = ["set -euo pipefail", f"mkdir -p {OUT_DIR}",
             f"python3 {EFF_DIR}/eff_prep.py {EFF_DIR}/spec.json > {OUT_DIR}/prep.json"]
    if arm.framework == "verl":
        lines += [f"cd {VERL_ROOT}", f"export VERL_FILE_LOGGER_PATH={OUT_DIR}/verl-metrics.jsonl",
                  f"export PYTHONPATH={EFF_DIR}",
                  # Modal: the trainer allocates NCCL host buffers with cuMem, the
                  # rollout-side receiver falls back to /dev/shm and cannot attach
                  # (eff01-bare-verl-async-d, NCCL_DEBUG=INFO); both use /dev/shm.
                  *(["export NCCL_CUMEM_HOST_ENABLE=0"] if arm.mode == "async" else []),
                  "exec python3 -m " + verl_bare_entry(arm.mode) + " " + shlex.join(
                      verl_bare_overrides(spec, arm.mode, run_id))]
    else:
        # model scripts derived from another (qwen3-4B-Instruct-2507.py) import Miles'
        # model_args_utils (miles/utils/external_utils), which is not on the scripts' path
        load = ("import importlib.util as u, sys; sys.path.insert(0, '{d}'); "
                f"sys.path.insert(0, '{MILES_ROOT}/miles/utils/external_utils'); "
                "s = u.spec_from_file_location('m', '{d}/{f}'); m = u.module_from_spec(s); "
                "s.loader.exec_module(m); print(m.model_args())").format(d=f"{MILES_ROOT}/scripts/models",
                                                                         f=spec.miles_model_args)
        runtime_env = json.dumps({"env_vars": {"PYTHONPATH": f"{MEGATRON_ROOT}/:{EFF_DIR}",
                                               "CUDA_DEVICE_MAX_CONNECTIONS": "1"}})
        lines += [f"cd {MILES_ROOT}", f"MODEL_ARGS=$(python3 -c {shlex.quote(load)})",
                  f"echo \"$MODEL_ARGS\" > {OUT_DIR}/model_args.txt",
                  f"ray start --head --node-ip-address 127.0.0.1 --num-gpus {c['total']} --num-cpus 32 "
                  "--disable-usage-stats --dashboard-host=127.0.0.1",
                  f"exec ray job submit --address=http://127.0.0.1:8265 --runtime-env-json={shlex.quote(runtime_env)} "
                  f"-- python3 {MILES_ROOT}/{miles_bare_script(arm.mode)} $MODEL_ARGS "
                  + shlex.join(miles_bare_argv(spec, arm.mode))]
    return "\n".join(lines) + "\n"


def plan(spec: EffSpec, arm_name: str, run_id: str) -> dict[str, Any]:
    """Everything one run of an arm needs (written to ``arm.json`` before the launch)."""
    arm = Arm.parse(arm_name)
    c = cards(arm.mode, spec.cards)
    doc: dict[str, Any] = {
        "arm": arm.name, "system": arm.system, "framework": arm.framework, "mode": arm.mode, "run_id": run_id,
        "spec": spec.to_dict(), "fingerprint": control_fingerprint(spec), "cards": c, "gpu": spec.gpu,
    }
    if arm.system == "bare":
        doc.update(harness="modal_eff", image=arm.framework, script=bare_script(spec, arm, run_id))
    elif arm.framework == "verl":
        # island_entry reads the rollout memory fraction from YETO_VERL_GPU_MEM (default 0.5)
        doc.update(harness="modal_eff", image="verl", env={"YETO_VERL_GPU_MEM": str(spec.mem_fraction)},
                   argv=yeto_verl_argv(spec, arm.mode, "/root/yeto-output/rl-island-0.jsonl"))
    else:
        doc.update(harness="miles_local", launcher_args=yeto_miles_launcher_args(spec, arm.mode, run_id))
    return doc


# ====================================================================== control check
def _override_map(overrides: list[str]) -> dict[str, str]:
    out = {}
    for o in overrides:
        key, _, value = o.partition("=")
        out[key.lstrip("+")] = value
    return out


def verl_knob_diff(spec: EffSpec, mode: str) -> dict[str, dict[str, str | None]]:
    """Hydra overrides where bare verl and yeto+verl differ (key -> {bare, yeto}).

    The yeto side is yeto's own translation (``build_overrides``, plus
    ``fully_async_run_overrides`` for async) of the same spec; paths, names and
    keys yeto handles in its own loop are compared too, so the list is the full
    set of differences, not only the interesting ones."""
    from yeto.rl.adapters.verl.config import VerlRunConfig, build_overrides
    from yeto.rl.adapters.verl.fully_async_round import fully_async_run_overrides

    cfg = VerlRunConfig(
        model_path=_model_path(spec), train_file=f"{DATA_DIR}/train.parquet", val_file=f"{DATA_DIR}/test.parquet",
        out_dir=OUT_DIR, lora_rank=spec.lora_r, lora_alpha=spec.lora_r, lr=spec.lr,
        groups_per_round=spec.groups_per_round, samples_per_group=spec.samples_per_group,
        max_prompt_len=spec.max_prompt_len, max_response_len=spec.max_response_len, seed=spec.seed,
        total_rounds=spec.rounds, temperature=spec.temperature, top_p=spec.top_p, tis_upper=spec.tis_upper,
        gpu_memory_utilization=spec.mem_fraction, reward_path=f"{EFF_DIR}/eff_reward.py",
        chat_template_kwargs=dict(spec.chat_template_kwargs), experiment_name="x")
    yeto = build_overrides(cfg)
    if mode == "async":
        yeto = fully_async_run_overrides(yeto, spec.max_policy_age, groups_per_round=spec.groups_per_round,
                                         rounds=spec.rounds)
    bare, ours = _override_map(verl_bare_overrides(spec, mode, "x")), _override_map(yeto)
    return {k: {"bare": bare.get(k), "yeto": ours.get(k)} for k in sorted(set(bare) | set(ours))
            if bare.get(k) != ours.get(k)}


def write_plan(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))
