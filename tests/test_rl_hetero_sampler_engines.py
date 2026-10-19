"""CPU tests of the engine-neutral sampler (S25 N2.1-N2.5, rl-hetero-sampler-engines).

No engine, card or network: the island runs on :class:`FakeEngine`; the SGLang
and vLLM adapters run on a fake process and a fake HTTP layer; the card tools
on recorded output.
"""

from __future__ import annotations

import json
import math
import signal
from pathlib import Path

import pytest

from yeto.hw import card_telemetry as ct
from yeto.hw.catalog import UnknownCardType, compat_group, compat_group_family
from yeto.launch.node_inventory import InventoryError, parse_inventory
from yeto.rl.adapters.sampler_engines import engine_class
from yeto.rl.adapters.sampler_engines.fake import FakeEngine
from yeto.rl.adapters.sampler_engines.server import ServerConfig
from yeto.rl.adapters.sampler_engines.sglang_server import SGLangOptions, SGLangServerEngine
from yeto.rl.adapters.sampler_engines.vllm_server import VllmOptions, VllmServerEngine
from yeto.rl.engine import telemetry
from yeto.rl.engine.buffer_payload import GroupWriter, decode_group
from yeto.rl.engine.engine_sampler import EngineSamplerConfig, PromptItem, build_engine_sampler
from yeto.rl.engine.loop_guard import LoopGuard
from yeto.rl.engine.offpolicy_source import SourceSample, offpolicy_source_event
from yeto.rl.engine.policy_release import PolicyReleaseStore
from yeto.rl.engine.resume import CheckpointStore
from yeto.rl.engine.sample_buffer import FileSampleBuffer
from yeto.rl.engine.sampler_checks import (
    GSM8K_FIXED_PROMPTS,
    LOAD_CHECK_FAILED_EVENT,
    PortReleaseLoader,
    PostLoadChecks,
    SamplerLoadFatal,
    count_release_experts,
)
from yeto.rl.engine.sampler_engine import (
    FINISH_LENGTH,
    FINISH_LOOP_ABORT,
    FINISH_STOP,
    Completion,
    EngineIdentity,
    Provenance,
    SamplerEngine,
    SamplerEngineError,
    SamplingConfigError,
    SamplingParams,
)
from yeto.rl.engine.trainer_lifecycle import SampleIndexEntry

ANSWERS = {p.prompt: f"The answer is {p.answer}." for p in GSM8K_FIXED_PROMPTS}
W4A8 = EngineIdentity(engine="fake", engine_version="0", device_family="npu", card_model="910B4",
                      precision="w4a8", quant_scheme="w4a8-int4-channel", topology="tp4")


# ---------------------------------------------------------------------- N2.1 catalog
def test_compat_group_carries_family_and_hbm():
    assert compat_group("H100") == "nvidia-h100"  # without HBM: unchanged
    assert compat_group("910B4", 61.9) == "ascend-910b4-64g"
    assert compat_group("910b4", 32) == "ascend-910b4-32g"
    assert compat_group("910B2", 64) == "ascend-910b2-64g"
    assert compat_group("910C", 128) == "ascend-910c-128g"
    assert compat_group("M890", 144.0) == "thead-m890-144g"
    assert compat_group("810E", 63.9) == "thead-810e-64g"  # sizes unknown: rounded
    assert compat_group_family("ascend-910b4-64g") == "npu"
    assert compat_group_family("thead-m890-144g") == "ppu"
    with pytest.raises(UnknownCardType, match="not a known variant"):
        compat_group("910B3", 32)


def _inventory(**node):
    base = {"name": "npu1", "host": "u@h", "device_family": "npu", "card": "910B4",
            "cards_per_node": 8, "hbm_gib": 64, "host_memory_gib": 1024, "image": "img"}
    return parse_inventory({"nodes": [{**base, **node}]})


def test_inventory_node_records_its_compat_group_and_refuses_unknown_hbm():
    (node,) = _inventory()
    assert node.compat_group == "ascend-910b4-64g"
    assert node.record()["compat_group"] == "ascend-910b4-64g"
    with pytest.raises(InventoryError, match="not a known variant"):
        _inventory(hbm_gib=48)


# ---------------------------------------------------------------------- N2.2 telemetry
NPU_SMI = """+---------------------------+---------------+----------------------------------+
| NPU   Name                | Health        | Power(W)  Temp(C)  Hugepages      |
| Chip                      | Bus-Id        | AICore(%) Memory-Usage(MB)        |
+===========================+===============+==================================+
| 0     910B4               | OK            | 92.5      47       0 / 0         |
| 0                         | 0000:C1:00.0  | 12        3161 / 65536           |
+===========================+===============+==================================+
| 1     910B4               | Warning       | 91.3      46       0 / 0         |
| 0                         | 0000:C2:00.0  | 0         0 / 0     3160 / 65536 |
+===========================+===============+==================================+
"""


def test_npu_smi_usage_has_util_hbm_power_temp_and_health_per_card():
    samples = ct.parse_npu_smi_usage(NPU_SMI)
    assert samples[0] == {"index": 0, "uuid": "0000:C1:00.0", "util_pct": 12.0,
                          "mem_used_mb": 3161.0, "mem_total_mb": 65536.0, "power_w": 92.5,
                          "temp_c": 47.0, "health": "OK"}
    assert (samples[1]["health"], samples[1]["mem_used_mb"]) == ("Warning", 3160.0)
    assert set(samples[0]) == set(ct.CARD_KEYS)
    with pytest.raises(ct.CardTelemetryError):
        ct.parse_npu_smi_usage("npu-smi: command not found")


def test_ppu_smi_query_rows():
    text = "0, PPU-1, 50, 1024, 98304, 300.5, 61\n1, PPU-2, [N/A], 0, 98304, [N/A], 60\n"
    samples = ct.parse_smi_query_csv(text)
    assert samples[0]["util_pct"] == 50.0 and samples[0]["temp_c"] == 61.0
    assert samples[1]["util_pct"] is None and samples[1]["power_w"] is None
    assert "--query-gpu=" + ",".join(ct.PPU_QUERY) in ct.PpuSmiCollector.ARGV


def test_resource_sampler_reads_the_family_collector(monkeypatch):
    events = []
    collector = ct.NpuSmiCollector(run=lambda argv: NPU_SMI)
    sampler = telemetry.ResourceSampler(lambda n, **f: events.append((n, f)), collector=collector,
                                        rss=lambda: 1)
    assert sampler.tick()
    name, fields = events[0]
    assert name == telemetry.RESOURCE_EVENT and fields["source"] == "npu-smi"
    assert fields["gpus"][1]["health"] == "Warning"
    assert sampler.take_peaks()["peak_gpu_mem_bytes"] == 3161 * 2**20
    # the device family from the environment selects the collector
    monkeypatch.setenv(telemetry.DEVICE_FAMILY_ENV, "ppu")
    assert telemetry.ResourceSampler(lambda *a, **k: None).source == "ppu-smi"


def test_missing_card_tool_writes_one_record_with_the_reason():
    def missing(argv):
        raise ct.CardTelemetryError("npu-smi: No such file or directory")

    events = []
    sampler = telemetry.ResourceSampler(lambda n, **f: events.append((n, f)), interval_s=0.001,
                                        collector=ct.NpuSmiCollector(run=missing)).start()
    sampler.stop()
    assert len(events) == 1
    assert events[0][1]["available"] is False and "No such file" in events[0][1]["reason"]
    assert events[0][1]["source"] == "npu-smi"


# ---------------------------------------------------------------------- N2.3 port types
def test_sampling_refuses_repetition_penalties_unless_allowed():
    with pytest.raises(SamplingConfigError, match="repetition_penalty"):
        SamplingParams(max_tokens=8, repetition_penalty=1.1).check()
    with pytest.raises(SamplingConfigError, match="presence_penalty"):
        SamplingParams(max_tokens=8, presence_penalty=0.5).check()
    allowed = SamplingParams(max_tokens=8, frequency_penalty=0.2, allow_penalties=True).check()
    assert allowed.digest() != SamplingParams(max_tokens=8).digest()
    assert SamplingParams(max_tokens=8).digest() == SamplingParams(max_tokens=8).digest()


def test_fake_engine_implements_the_port():
    assert isinstance(FakeEngine(), SamplerEngine)
    assert engine_class("fake") is FakeEngine
    with pytest.raises(ValueError, match="unknown sampler engine"):
        engine_class("trt")


def test_loop_guard_cuts_a_repeating_completion_and_keeps_its_logprobs():
    guard = LoopGuard(max_period=3, repeats=4)
    ids = (5, 6, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3)
    completion = Completion(token_ids=ids, logprobs=tuple(-0.1 * i for i in range(len(ids))),
                            text="loop", finish_reason=FINISH_LENGTH)
    cut = guard.apply(completion)
    assert cut.finish_reason == FINISH_LOOP_ABORT
    assert cut.token_ids == ids[:14] and cut.logprobs == completion.logprobs[:14]
    plain = Completion(token_ids=(1, 2, 3, 4), logprobs=(0.0,) * 4, text="x", finish_reason=FINISH_STOP)
    assert guard.apply(plain) is plain


# ---------------------------------------------------------------------- N2.4 provenance
def test_index_lines_carry_provenance_and_old_lines_read_as_unknown(tmp_path):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    old = SampleIndexEntry("old", 0, buffer.write_payload("old", b"{}"), tokens=1)
    buffer.append(old)
    record = Provenance.of(W4A8, sampler_id="npu1", sampling=SamplingParams(max_tokens=8)).to_record()
    buffer.append(SampleIndexEntry("new", 1, buffer.write_payload("new", b"{}"), tokens=1,
                                   provenance=record))
    lines = [json.loads(line) for line in
             (tmp_path / "sample-buffer" / "index.jsonl").read_text().splitlines()]
    assert "provenance" not in lines[0] and lines[1]["provenance"]["card_model"] == "910B4"
    reader = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    reader.refresh()
    first, second = (s.entry for s in reader.select("t", 2))
    assert Provenance.from_record(first.provenance) == Provenance.unknown()
    assert Provenance.from_record(second.provenance).quant_scheme == "w4a8-int4-channel"
    assert Provenance.from_record(second.provenance).key() == "npu/910B4/fake@0/w4a8/tp4"


# ---------------------------------------------------------------------- N2.5 per-source metrics
def _npu_ppu(record_npu, record_ppu):
    small = SourceSample(record_npu, [-1.0, -2.0], [-1.0, -2.01], FINISH_STOP)
    shifted = SourceSample(record_ppu, [-1.0, -1.0, -1.0], [-2.0, -1.0, -1.0], FINISH_LENGTH)
    return [small, small, shifted, SourceSample(None, [-1.0], [-1.0])]


def test_offpolicy_source_groups_recorded_backend_values_by_source():
    npu = Provenance.of(W4A8, sampler_id="a", sampling=SamplingParams(max_tokens=8)).to_record()
    ppu_identity = EngineIdentity(engine="sglang", engine_version="0.5.17", device_family="ppu",
                                  card_model="810E", precision="mxfp4", quant_scheme="MXFP4_EXPERTS_RTN",
                                  topology="tp8")
    ppu = Provenance.of(ppu_identity, sampler_id="b", sampling=SamplingParams(max_tokens=8)).to_record()
    event = offpolicy_source_event(_npu_ppu(npu, ppu), rollout_id=3, trainer_version=7, backend="verl")
    assert event["consumed"] == 4
    sources = {s["key"]: s for s in event["sources"]}
    assert sum(s["samples"] for s in sources.values()) == event["consumed"]
    assert set(sources) == {"npu/910B4/fake@0/w4a8/tp4", "ppu/810E/sglang@0.5.17/mxfp4/tp8",
                            "unknown/unknown/unknown@unknown/unknown/unknown"}
    p = sources["ppu/810E/sglang@0.5.17/mxfp4/tp8"]
    assert p["tokens"] == 3 and p["truncated_share"] == 1.0
    assert math.isclose(p["ratio_max"], math.e) and math.isclose(p["tis_clipfrac"], 1 / 3)
    assert math.isclose(p["abs_diff_mean"], 1 / 3)
    n = sources["npu/910B4/fake@0/w4a8/tp4"]
    assert n["tis_clipfrac"] == 0.0 and math.isclose(n["abs_diff_p99"], 0.01, abs_tol=1e-9)
    assert n["finish"] == {"stop": 2} and n["by_length"]["0-255"]["samples"] == 2


# ---------------------------------------------------------------------- post-load checks
def _w4a8_release(releases, version, *, experts=4, scheme="w4a8-int4-channel", recorded=None):
    def write(directory: Path) -> None:
        names = {f"model.layers.0.mlp.experts.{e}.down_proj.weight": "a.safetensors"
                 for e in range(experts)}
        names["model.embed_tokens.weight"] = "a.safetensors"
        (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": names}))
        (directory / "a.safetensors").write_bytes(b"x")

    derived = {"scheme": scheme, "version": version}
    if recorded is not None:
        derived["expert_matrices"] = recorded
    return releases.release(version, f"h{version}", write, derived_from=derived)


def _loader(engine, events, **checks):
    return PortReleaseLoader(engine, PostLoadChecks(**{"expected_expert_matrices": 4, **checks}),
                             lambda n, **f: events.append((n, f)), max_consecutive_failures=2)


def test_release_expert_count_reads_the_index(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    release = _w4a8_release(releases, 1, experts=3)
    assert count_release_experts(releases.resolve(release)) == 3
    assert count_release_experts(tmp_path) is None


def test_loader_serves_a_release_only_after_all_checks_pass(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=4)
    events = []
    loader = _loader(engine, events, max_probe_jump=0.5)
    loader.start(None, 0)
    release = _w4a8_release(releases, 1)
    loader(release, releases.resolve(release))
    assert loader.ready and engine.served_version() == 1
    checked = [f for n, f in events if n == "rl_sampler_load_checked"]
    assert [c["reload_mode"] for c in checked] == ["restart", "in_place"]
    assert {c["name"] for c in checked[1]["checks"]} == {"expert_count", "fixed_prompts",
                                                         "logprob_probe", "served_version"}
    assert checked[1]["checks"][2]["against"] == "previous"


@pytest.mark.parametrize("fault, step", [
    ("silent_skip", "expert_count"),  # the engine loaded fewer experts (A-INF item 1)
    ("wrong_answers", "fixed_prompts"),
    ("probe_jump", "logprob_probe"),
])
def test_a_failed_check_marks_the_engine_not_ready_and_restarts_next_time(tmp_path, fault, step):
    from yeto.rl.engine.policy_release import ReleaseNotLoaded

    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=4)
    events = []
    loader = _loader(engine, events, max_probe_jump=0.5)
    loader.start(None, 0)
    release = _w4a8_release(releases, 1)
    if fault == "silent_skip":
        engine.expert_matrices = 0
    elif fault == "wrong_answers":
        engine.answers = {}
    elif fault == "probe_jump":
        engine.probe_shift = 2.0
    with pytest.raises(ReleaseNotLoaded, match=step):
        loader(release, releases.resolve(release))
    assert not loader.ready
    (failed,) = [f for n, f in events if n == LOAD_CHECK_FAILED_EVENT]
    assert failed["step"] == step and failed["failures"] == 1
    with pytest.raises(SamplerLoadFatal):  # second failure in a row: the island stops
        loader(release, releases.resolve(release))


def test_a_release_with_the_wrong_expert_count_keeps_the_served_weights(tmp_path):
    """The release is refused before the engine is touched: v0 stays served and ready."""
    from yeto.rl.engine.policy_release import ReleaseNotLoaded

    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=4)
    events = []
    loader = _loader(engine, events)
    loader.start(None, 0)
    bad = _w4a8_release(releases, 1, experts=3)
    with pytest.raises(ReleaseNotLoaded, match="release_expert_count") as refused:
        loader(bad, releases.resolve(bad))
    assert refused.value.reason == "release_refused:release_expert_count"
    assert loader.ready and engine.served_version() == 0
    (failed,) = [f for n, f in events if n == LOAD_CHECK_FAILED_EVENT]
    assert failed["engine_touched"] is False and failed["kept_version"] == 0
    assert failed["detail"] == {"in_release": 3, "expected": 4}

    good = _w4a8_release(releases, 2)  # the engine was never touched: an in-place load follows
    loader(good, releases.resolve(good))
    assert loader.ready and engine.served_version() == 2 and loader.failures == 0
    checked = [f for n, f in events if n == "rl_sampler_load_checked"]
    assert checked[-1]["reload_mode"] == "in_place"

    with pytest.raises(ReleaseNotLoaded):
        loader(bad, releases.resolve(bad))
    with pytest.raises(SamplerLoadFatal):  # still counted: a run that only publishes bad releases stops
        loader(bad, releases.resolve(bad))


def test_an_engine_error_inside_a_check_is_a_failed_check(tmp_path):
    from yeto.rl.engine.policy_release import ReleaseNotLoaded
    from yeto.rl.engine.sampler_engine import SamplerEngineError

    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=4)
    events = []
    loader = _loader(engine, events)
    loader.start(None, 0)

    def broken_score(prompt, continuation):
        raise SamplerEngineError("HTTP 500")

    engine.score = broken_score
    release = _w4a8_release(releases, 1)
    with pytest.raises(ReleaseNotLoaded, match="checks"):
        loader(release, releases.resolve(release))
    assert not loader.ready  # the engine now holds v1 weights that were never checked
    assert [f["step"] for n, f in events if n == LOAD_CHECK_FAILED_EVENT] == ["checks"]


def test_probe_of_a_load_that_failed_another_check_is_not_the_next_base(tmp_path):
    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=4)
    loader = _loader(engine, [], max_probe_jump=0.5)
    loader.start(None, 0)
    base = list(loader.checks._last_probe)
    engine.answers = {}  # fixed prompts fail; the probe itself passes
    release = _w4a8_release(releases, 1)
    with pytest.raises(Exception, match="fixed_prompts"):
        loader(release, releases.resolve(release))
    assert loader.checks._last_probe == base


def test_engine_that_reports_no_count_needs_memory_or_an_explicit_waiver(tmp_path):
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=None)
    checks = PostLoadChecks(expected_expert_matrices=4)
    assert checks.expert_count(engine.__class__(identity=W4A8, expert_matrices=None), 4)["passed"] is False
    engine.weight_gib = 20.8
    assert PostLoadChecks(expected_weight_gib_per_card=20.0).expert_count(engine, 4)["passed"]
    assert not PostLoadChecks(expected_weight_gib_per_card=15.0).expert_count(engine, 4)["passed"]
    waived = PostLoadChecks(allow_unverified_count=True).expert_count(
        FakeEngine(expert_matrices=None), 4)
    assert waived["passed"] and waived["unverified"]


def test_scheme_mismatch_keeps_the_served_version(tmp_path):
    from yeto.rl.engine.policy_release import ReleaseNotLoaded

    releases = PolicyReleaseStore(CheckpointStore(tmp_path))
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=4)
    loader = _loader(engine, [])
    loader.start(None, 0)
    bf16 = releases.release(1, "h1", lambda d: (d / "w").write_bytes(b"x"))
    with pytest.raises(ReleaseNotLoaded) as info:
        loader(bf16, releases.resolve(bf16))
    assert info.value.reason == "scheme_mismatch"
    assert engine.served_version() == 0 and loader.ready and loader.failures == 0


# ---------------------------------------------------------------------- island end to end
def test_engine_sampler_island_writes_groups_with_version_and_provenance(tmp_path):
    store = CheckpointStore(tmp_path)
    releases = PolicyReleaseStore(store)
    buffer = FileSampleBuffer(store, batch_size=1)
    engine = FakeEngine(identity=W4A8, answers=ANSWERS, expert_matrices=4, max_model_len=64)
    events = []
    config = EngineSamplerConfig(sampler_id="npu1", num_rollouts=2, max_backlog_groups=100,
                                 sampling=SamplingParams(max_tokens=6, n=2), max_prompt_tokens=32)

    def prompts(rollout_id):
        if rollout_id == 0:  # the trainer releases v1 during round 0
            _w4a8_release(releases, 1)
        return [PromptItem(f"p{rollout_id}", f"prompt {rollout_id}", label="x")]

    island = build_engine_sampler(config, engine=engine, releases=releases, buffer=buffer,
                                  emit=lambda n, **f: events.append((n, f)), prompts=prompts,
                                  reward=lambda item, c: 1.0,
                                  checks=PostLoadChecks(expected_expert_matrices=4),
                                  sleep=lambda s: None)
    assert island.run() == 2
    reader = FileSampleBuffer(store, batch_size=1)
    reader.refresh()
    entries = [s.entry for s in reader.select("t", 10)]
    assert [e.behavior_version for e in entries] == [0, 1]
    assert all(Provenance.from_record(e.provenance).card_model == "910B4" for e in entries)
    samples = decode_group(reader.read_payload(entries[0].payload_ref))
    assert len(samples) == 2 and samples[0]["metadata"]["finish_reason"] in (FINISH_STOP, FINISH_LENGTH)
    assert len(samples[0]["rollout_log_probs"]) == samples[0]["response_length"]
    names = [n for n, _ in events]
    assert names[0] == "rl_sampler_start" and "rl_sampler_engine_round" in names
    start = events[0][1]
    assert start["sampling"]["repetition_penalty"] == 1.0 and start["provenance"]["precision"] == "w4a8"


def test_island_refuses_max_tokens_the_engine_cannot_serve(tmp_path):
    store = CheckpointStore(tmp_path)
    config = EngineSamplerConfig(sampler_id="s", num_rollouts=1, max_backlog_groups=1,
                                 sampling=SamplingParams(max_tokens=16384), max_prompt_tokens=1024)
    with pytest.raises(SamplerEngineError, match="max_model_len 8192"):
        build_engine_sampler(config, engine=FakeEngine(max_model_len=8192),
                             releases=PolicyReleaseStore(store), buffer=FileSampleBuffer(store, batch_size=1),
                             emit=lambda *a, **k: None, prompts=lambda r: [], reward=lambda i, c: 0.0)


def test_group_writer_puts_provenance_in_the_entry(tmp_path):
    buffer = FileSampleBuffer(CheckpointStore(tmp_path), batch_size=1)
    sample = {"response_length": 1, "rollout_log_probs": [-0.5], "metadata": {"weight_version": "3"}}
    entry = GroupWriter(buffer, "s").write([sample], group_key="g", provenance={"engine": "vllm"})
    assert entry.behavior_version == 3 and entry.provenance == {"engine": "vllm"}


# ---------------------------------------------------------------------- server adapters
class _Process:
    def __init__(self, exit_on_term=0):
        self.code = None
        self.exit_on_term = exit_on_term
        self.signals = []

    def poll(self):
        return self.code

    def send_signal(self, sig):
        self.signals.append(sig)
        self.code = self.exit_on_term

    def kill(self):
        self.code = -9

    def wait(self, timeout=None):
        return self.code


class _Http:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, method, url, body, timeout):
        path = url.split(":30000", 1)[1] if ":30000" in url else url
        self.calls.append((method, path, body))
        handler = self.routes.get((method, path))
        return (404, None) if handler is None else (200, handler(body) if callable(handler) else handler)


def _config(tmp_path, **kw):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    values = dict(base_dir=base, card_model="810E", device_family="ppu", engine_version="0.5.17",
                  precision="mxfp4", quant_scheme="MXFP4_EXPERTS_RTN", topology="tp8", max_model_len=32768)
    values.update(kw)
    return ServerConfig(**values)


def _sglang_generate(body):
    texts = body["text"] if isinstance(body["text"], list) else [body["text"]]
    if body["sampling_params"]["max_new_tokens"] == 0:  # teacher-forced score
        return [{"meta_info": {"input_token_logprobs": [[None, 1, None]] + [[-0.5, 2, None]] * (len(t) - 1)}}
                for t in texts]
    return [{"text": f"out {t}", "meta_info": {"output_token_logprobs": [[-0.1, 7, None], [-0.2, 8, None]],
                                              "finish_reason": {"type": "length"}}} for t in texts]


def test_sglang_adapter_argv_env_generate_and_score(tmp_path):
    spawned = []
    http = _Http({("GET", "/health"): {}, ("POST", "/generate"): _sglang_generate})
    engine = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: spawned.append((a, e)) or _Process(134),
                                http=http, sleep=lambda s: None)
    report = engine.start(None, 0)
    argv, env = spawned[0]
    assert argv[:3] == ["python", "-m", "sglang.launch_server"] and "--tp" in argv
    assert env["MX_PATCH"] == "1" and report.detail["env"] == {"MX_PATCH": "1"}
    groups = engine.generate(["a", "b"], SamplingParams(max_tokens=2, n=2))
    assert [len(g) for g in groups] == [2, 2]
    assert groups[0][0].token_ids == (7, 8) and groups[0][0].finish_reason == FINISH_LENGTH
    sent = [b for m, p, b in http.calls if p == "/generate"][0]
    assert sent["text"] == ["a", "a", "b", "b"] and sent["sampling_params"]["repetition_penalty"] == 1.0
    assert engine.score("abc", "de") == [-0.5, -0.5]
    engine.stop()  # TP8 exits 134 (pcclCommAbort) after a stop request: normal
    assert engine.last_exit["code"] == 134 and engine.last_exit["normal"]


def test_sglang_exit_134_while_serving_is_a_failure(tmp_path):
    process = _Process()
    engine = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: process,
                                http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    engine.start(None, 0)
    process.code = 134
    with pytest.raises(SamplerEngineError, match="exited with code 134 while serving"):
        engine.generate(["a"], SamplingParams(max_tokens=2))
    assert engine.served_version() is None


def test_sglang_refuses_dp8_with_nextn_packed_names_and_dense_format(tmp_path):
    with pytest.raises(SamplerEngineError, match="NEXTN"):
        SGLangServerEngine(_config(tmp_path, topology="dp8"), SGLangOptions(nextn=True))
    release = tmp_path / "r1"
    release.mkdir()
    (release / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"model.layers.0.mlp.experts.0.down_proj.weight_packed": "a"}}))
    engine = SGLangServerEngine(_config(tmp_path), spawn=lambda a, e: _Process(),
                                http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    with pytest.raises(SamplerEngineError, match="weight_packed"):
        engine.start(release, 1)
    patched = SGLangServerEngine(_config(tmp_path), SGLangOptions(renames_weight_packed=True),
                                 spawn=lambda a, e: _Process(), http=_Http({("GET", "/health"): {}}),
                                 sleep=lambda s: None)
    assert patched.start(release, 1).reload_mode == "restart"
    (release / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "compressed-tensors",
                                                                            "format": "dense"}}))
    with pytest.raises(SamplerEngineError, match="dense"):
        patched.start(release, 1)


def test_sglang_dp8_warms_up_batch_sizes_before_the_checks(tmp_path):
    http = _Http({("GET", "/health"): {}, ("POST", "/generate"): _sglang_generate})
    engine = SGLangServerEngine(_config(tmp_path, topology="dp8"), SGLangOptions(warmup_batch_sizes=(1, 4)),
                                spawn=lambda a, e: _Process(), http=http, sleep=lambda s: None)
    report = engine.start(None, 0)
    assert report.detail["warmup_batch_sizes"] == [1, 4]
    assert [len(b["text"]) for m, p, b in http.calls if p == "/generate"] == [1, 4]


def test_start_timeout_kills_the_server_and_names_the_hang(tmp_path):
    clock = iter(range(0, 10_000, 100))
    process = _Process()
    engine = VllmServerEngine(_vllm(tmp_path, start_timeout_s=300),
                              spawn=lambda a, e: process, http=_Http({}), clock=lambda: next(clock),
                              sleep=lambda s: None)
    with pytest.raises(SamplerEngineError, match="HCCL hang"):
        engine.start(None, 0)
    assert process.code == -9


def _vllm(tmp_path, **kw):
    cfg = _config(tmp_path, device_family="npu", card_model="910B4", precision="w4a8",
                  quant_scheme="w4a8-int4-channel", topology="tp4", max_model_len=8192, **kw)
    (cfg.base_dir / "config.json").write_text(json.dumps(
        {"quantization_config": {"quant_method": "compressed-tensors", "format": "int-quantized"}}))
    return cfg


def test_vllm_adapter_restarts_on_each_release_and_parses_completions(tmp_path):
    spawned = []

    def completions(body):
        if body.get("echo"):
            n = len(body["prompt"])
            return {"choices": [{"index": 0, "logprobs": {"token_logprobs": [None] + [-0.25] * (n - 1) + [-1.0]}}]}
        return {"choices": [{"index": i, "text": "t", "finish_reason": "stop",
                             "logprobs": {"tokens": ["token_id:4", "token_id:5"],
                                          "token_logprobs": [-0.3, -0.4]}}
                            for i in range(len(body["prompt"]) * body["n"])]}

    http = _Http({("GET", "/health"): {}, ("POST", "/v1/completions"): completions})
    engine = VllmServerEngine(_vllm(tmp_path), VllmOptions(max_num_seqs=32),
                              spawn=lambda a, e: spawned.append((a, e)) or _Process(), http=http,
                              sleep=lambda s: None)
    engine.start(None, 0)
    release = tmp_path / "r1"
    release.mkdir()
    (release / "config.json").write_text(json.dumps({"quantization_config": {"format": "int-quantized"}}))
    assert engine.load(release, 1).reload_mode == "restart" and engine.served_version() == 1
    assert len(spawned) == 2 and spawned[0][1]["VLLM_ASCEND_W4A8_CPU_STAGE"] == "1"
    assert "--max-num-seqs" in spawned[0][0]
    (group,) = engine.generate(["p"], SamplingParams(max_tokens=4))
    assert group[0].token_ids == (4, 5) and group[0].logprobs == (-0.3, -0.4)
    assert engine.score("ab", "cd") == [-0.25, -0.25]


def test_vllm_refuses_eager_pack_quantized_in_place_and_low_host_memory(tmp_path):
    with pytest.raises(SamplerEngineError, match="enforce-eager"):
        VllmServerEngine(_vllm(tmp_path, extra_args=("--enforce-eager",)))
    assert VllmServerEngine(_vllm(tmp_path, extra_args=("--enforce-eager",)), VllmOptions(allow_eager=True))
    with pytest.raises(SamplerEngineError, match="in-place reload"):
        VllmServerEngine(_vllm(tmp_path, in_place_reload=True))
    release = tmp_path / "r1"
    release.mkdir()
    (release / "config.json").write_text(json.dumps({"quantization_config": {"format": "pack-quantized"}}))
    engine = VllmServerEngine(_vllm(tmp_path), spawn=lambda a, e: _Process(),
                              http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    with pytest.raises(SamplerEngineError, match="pack-quantized"):
        engine.start(release, 1)
    low = VllmServerEngine(_vllm(tmp_path, min_host_free_gib=610), spawn=lambda a, e: _Process(),
                           http=_Http({("GET", "/health"): {}}), sleep=lambda s: None,
                           host_free_gib=lambda: 400.0)
    with pytest.raises(SamplerEngineError, match="610"):
        low.start(None, 0)


def test_stop_sends_sigterm(tmp_path):
    process = _Process()
    engine = VllmServerEngine(_vllm(tmp_path), spawn=lambda a, e: process,
                              http=_Http({("GET", "/health"): {}}), sleep=lambda s: None)
    engine.start(None, 0)
    engine.stop()
    assert process.signals == [signal.SIGTERM] and engine.last_exit["normal"]
    engine.stop()  # idempotent
