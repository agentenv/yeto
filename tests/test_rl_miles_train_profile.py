"""E1: --rl-profile-trainer is observation only and reaches the trainer."""

from __future__ import annotations

import gc
import json
import time

from yeto.rl.adapters.miles import train_profile


def test_profiled_returns_the_call_result_and_prints_one_record(capsys):
    def step(x, *, y):
        junk = [[i] for i in range(20000)]
        gc.collect()
        time.sleep(0.03)
        return x + y, len(junk)

    assert train_profile.profiled(step, 2, y=3) == (5, 20000)
    (line,) = [l for l in capsys.readouterr().out.splitlines() if l.startswith("yeto train profile ")]
    record = json.loads(line[len("yeto train profile "):])
    assert record["wall_s"] >= 0.03 and record["gc_pause_s"] > 0 and record["gc_collections"]
    assert record["samples"] >= 1 and any("step" in label for label, _ in record["main_top"])


def test_the_flag_reaches_the_launcher_args_and_the_eval_spec():
    from tools.eval.efficiency import spec

    sp = spec.get_spec("qwen3-0.6b", checkpoint="every-round", profile_trainer=True)
    assert " --rl-profile-trainer" in spec.yeto_miles_launcher_args(sp, "async", "r1")
    assert spec.control_fingerprint(sp) == spec.control_fingerprint(spec.get_spec("qwen3-0.6b"))


def test_the_flag_is_forwarded_to_the_island_and_set_on_the_miles_args(monkeypatch):
    import argparse

    from yeto.rl.engine import phase_profile

    # the sampler would stay on for the rest of the test process and print on every
    # driver phase (it broke the JSON output of test_rl_reconfig_e1 in the full suite)
    monkeypatch.setattr(phase_profile, "enable", lambda: None)
    from types import SimpleNamespace

    from yeto.launch import rl_ports
    from yeto.rl.adapters.miles import island_entry
    from yeto.rl.engine.run_flags import ISLAND, add_rl_flags

    _prelude, flags = rl_ports._ports_infra_flags(argparse.Namespace(rl_profile_trainer=True))
    assert " --rl-profile-trainer" in flags
    parser = argparse.ArgumentParser()
    add_rl_flags(parser, ISLAND)
    for action in parser._actions:  # parse just this flag
        action.required = False
    island_args, _ = parser.parse_known_args(["--rl-profile-trainer"])
    miles_args = SimpleNamespace()
    island_entry.apply_ports_infra_switches(island_args, miles_args)
    assert miles_args.yeto_rl_profile_trainer is True


def test_phase_profile_prints_each_finished_phase(capsys, monkeypatch):
    from yeto.rl.engine import phase_profile

    monkeypatch.setitem(phase_profile._STATE, "thread", None)
    phase_profile.on_phase("train", 0)  # disabled: nothing
    import threading

    phase_profile.enable()  # from this thread; the phases come from another one

    def busy_driver():
        phase_profile.on_phase("train", 1)
        end = time.perf_counter() + 0.05
        while time.perf_counter() < end:
            sum(range(1000))
        phase_profile.on_phase("sync", 1)

    driver = threading.Thread(target=busy_driver)
    driver.start()
    driver.join()
    (line,) = [l for l in capsys.readouterr().out.splitlines() if l.startswith("yeto phase profile ")]
    record = json.loads(line[len("yeto phase profile "):])
    assert record["phase"] == "train" and record["rollout_id"] == 1 and record["wall_s"] >= 0.05
    assert record["samples"] >= 1 and any("busy_driver" in label for label, _ in record["top"]), record
    sampler = phase_profile._STATE["thread"]
    sampler.phase = None
    monkeypatch.setitem(phase_profile._STATE, "thread", None)  # off for later tests
