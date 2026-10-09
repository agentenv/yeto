"""secret-handling-hardening: legacy HELLO HMAC, head-pinned island contract,
token boundary for learner islands, and the island start-up credential check."""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import subprocess
import time

import pytest
import torch

from yeto import island_credential_guard as guard
from yeto.protocol import (
    DTYPE_F32,
    HELLO_MAC_DOMAIN,
    SyncerClient,
    layout_fingerprint,
    seal_hello,
)
from yeto.rl.core import StrictRlInvariantError
from yeto.rl.engine.backend_identity import session_contract_hash
from yeto.tensor_io import pack_tensor

from test_rl_integration import _layout, _port, _wait_item, syncer_binary  # noqa: F401

GOOD = "3b9e60e5a2c554a2826f6cef83fdeecabfc1723d6663c4d5fd23b01e08a555e2"
BAD = "9d5696a3d3b6e6d802115ef3deb970b5e4d9206d1751d71f9075a849d2909f8d"


# --------------------------------------------------------------------------- D1 HELLO HMAC

def test_seal_hello_appends_domain_separated_hmac():
    body = b"hello-body"
    assert seal_hello(body, None) == body
    sealed = seal_hello(body, b"k")
    assert sealed[:-32] == body
    assert sealed[-32:] == hmac.new(b"k", HELLO_MAC_DOMAIN + body, hashlib.sha256).digest()


def _syncer(binary, port, tmp_path, *, key=None, allow_unauth=False, extra=(), learners=1):
    env = {k: v for k, v in os.environ.items()
           if k not in ("YETO_ISLAND_HMAC_KEY", "YETO_SYNCER_ALLOW_UNAUTHENTICATED")}
    if key is not None:
        env["YETO_ISLAND_HMAC_KEY"] = key
    argv = [str(binary), "--port", str(port), "--learners", str(learners), "--quorum", str(learners),
            "--grace-ms", "0", "--pipeline", "1", "--sync-interval-steps", "0",
            "--delta-correction", "none", "--total-steps", "1", "--outer-lr", "1",
            "--outer-momentum", "0", "--quorum-timeout-s", "10",
            "--checkpoint-path", str(tmp_path / "state.ckpt"), "--checkpoint-every", "1",
            "--max-base-lag", "0", "--learner-weight", "equal", *extra]
    if allow_unauth:
        argv.append("--allow-unauthenticated-islands")
    return subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)


def _client(port, learner_id=0, *, key, identity=GOOD):
    contract = session_contract_hash(layout_fingerprint(_layout()), identity)
    client = SyncerClient(("127.0.0.1", port), learner_id, _layout(), dtype=DTYPE_F32, num_streams=0,
                          connect_timeout=10, session_contract_hash=contract, max_reconnects=0,
                          hmac_key=key)
    client.start()
    return client


def _refused(client, timeout=10) -> StrictRlInvariantError:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            client.check_health()
        except StrictRlInvariantError as error:
            return error
        time.sleep(0.05)
    raise AssertionError("the syncer did not refuse this island")


def _stop(process):
    process.kill()
    return process.communicate(timeout=10)[0]


def test_legacy_syncer_without_key_refuses_to_start(syncer_binary, tmp_path):
    process = _syncer(syncer_binary, _port(), tmp_path)
    out = process.communicate(timeout=30)[0]
    assert process.returncode != 0
    assert "legacy mode requires an island HMAC key" in out


def test_legacy_hello_with_right_key_is_admitted_wrong_key_refused(syncer_binary, tmp_path):
    port = _port()
    process = _syncer(syncer_binary, port, tmp_path, key="s3cret-key")
    clients = []
    try:
        intruder = _client(port, 0, key=b"wrong-key")
        clients.append(intruder)
        error = _refused(intruder)
        assert error.metric == "island_auth_failed"
        assert "s3cret" not in str(error)
        no_key = _client(port, 0, key=b"")
        clients.append(no_key)
        assert _refused(no_key).metric == "island_auth_failed"
        good = _client(port, 0, key=b"s3cret-key")
        clients.append(good)
        good.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
        assert _wait_item(good.drain_updates).version == 0
        assert process.poll() is None, "the syncer must keep running after refusals"
    finally:
        for c in clients:
            c.close()
        _stop(process)


# --------------------------------------------------------------------------- D2 head-pinned contract

def test_pinned_contract_refuses_a_negative_island_that_connects_first(syncer_binary, tmp_path):
    port = _port()
    process = _syncer(syncer_binary, port, tmp_path, key="k3y",
                      extra=("--expected-island-contract", GOOD))
    clients = []
    try:
        negative = _client(port, 0, key=b"k3y", identity=BAD)
        clients.append(negative)
        error = _refused(negative)
        assert error.metric == "layout_hash_mismatch" and "the head pins" in str(error)
        good = _client(port, 0, key=b"k3y", identity=GOOD)
        clients.append(good)
        good.send_init(0, pack_tensor(torch.zeros(2), DTYPE_F32))
        assert _wait_item(good.drain_updates).version == 0
    finally:
        for c in clients:
            c.close()
        _stop(process)


def test_bad_pinned_contract_value_is_refused(syncer_binary, tmp_path):
    process = _syncer(syncer_binary, _port(), tmp_path, key="k",
                      extra=("--expected-island-contract", "XYZ"))
    out = process.communicate(timeout=30)[0]
    assert process.returncode != 0 and "64 lowercase hex" in out


# --------------------------------------------------------------------------- launcher

def _rl_args(**kw):
    import argparse

    base = dict(training_mode="rl", rl_island_scheduling="legacy", rl_island_contract_sha256=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_launcher_generates_one_legacy_key_per_run(monkeypatch):
    from yeto import launcher

    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY", raising=False)
    first = launcher.island_hmac_secret(_rl_args())
    second = launcher.island_hmac_secret(_rl_args())
    key = first["YETO_ISLAND_HMAC_KEY"]
    assert len(key) == 64 and first == second
    assert launcher.island_hmac_secret(_rl_args(training_mode="sft")) == {}


def test_launcher_elastic_still_needs_a_user_key(monkeypatch):
    from yeto import launcher

    monkeypatch.delenv("YETO_ISLAND_HMAC_KEY", raising=False)
    with pytest.raises(ValueError, match="YETO_ISLAND_HMAC_KEY"):
        launcher.island_hmac_secret(_rl_args(rl_island_scheduling="elastic"))


def test_launcher_passes_the_pinned_contract_to_the_syncer():
    from yeto import launcher

    assert launcher._syncer_expected_contract(_rl_args()) == ""
    assert launcher._syncer_expected_contract(_rl_args(rl_island_contract_sha256=GOOD)) == (
        f" --expected-island-contract {GOOD}")


def test_cli_validates_the_pinned_contract():
    from yeto import cli

    assert cli._sha256_hex_arg(GOOD) == GOOD
    with pytest.raises(Exception):
        cli._sha256_hex_arg("ABC")


# --------------------------------------------------------------------------- D3 token boundary

def test_secret_envs_are_split_out_of_plain_envs():
    from yeto import launcher

    plain, secrets = launcher.split_secret_envs(
        {"HF_TOKEN": "h", "WANDB_API_KEY": "w", "LEARNER_ID": "0",
         "YETO_SANDBOX_MODAL_TOKEN_SECRET": "s"})
    assert plain == {"LEARNER_ID": "0"}
    assert secrets == {"HF_TOKEN": "h", "WANDB_API_KEY": "w", "YETO_SANDBOX_MODAL_TOKEN_SECRET": "s"}


def test_main_modal_token_is_not_a_harness_passthrough():
    from yeto import launcher

    assert not set(launcher.HARNESS_PASSTHROUGH_ENV) & launcher.CLOUD_CREDENTIAL_ENV_NAMES
    with pytest.raises(ValueError, match="YETO_SANDBOX_MODAL_TOKEN_ID"):
        launcher.require_sandbox_modal_token({"MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b"})
    launcher.require_sandbox_modal_token(
        {"YETO_SANDBOX_MODAL_TOKEN_ID": "a", "YETO_SANDBOX_MODAL_TOKEN_SECRET": "b"})


def test_sandbox_client_uses_only_the_sandbox_token(monkeypatch):
    from yeto.rl.harness.codex import tb2_provider

    assert tb2_provider.sandbox_modal_client({"MODAL_TOKEN_ID": "a", "MODAL_TOKEN_SECRET": "b"}) is None
    seen = {}

    class FakeClient:
        @staticmethod
        def from_credentials(token_id, token_secret):
            seen["args"] = (token_id, token_secret)
            return "client"

    import sys
    import types

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(Client=FakeClient))
    env = {"YETO_SANDBOX_MODAL_TOKEN_ID": "sid", "YETO_SANDBOX_MODAL_TOKEN_SECRET": "ssec"}
    assert tb2_provider.sandbox_modal_client(env) == "client"
    assert seen["args"] == ("sid", "ssec")


# --------------------------------------------------------------------------- D4 island start-up check

def test_guard_names_match_the_launcher_list():
    from yeto import launcher

    assert guard.CLOUD_CREDENTIAL_ENV_NAMES == launcher.CLOUD_CREDENTIAL_ENV_NAMES


def test_guard_errors_on_env_credentials_without_printing_values(tmp_path):
    with pytest.raises(guard.IslandCloudCredentialError) as info:
        guard.check_island_credentials({"MODAL_TOKEN_SECRET": "as-vsec1"}, home=tmp_path,
                                       stream=io.StringIO())
    assert "MODAL_TOKEN_SECRET" in str(info.value) and "vsec1" not in str(info.value)


def test_guard_allow_env_downgrades_to_warning(tmp_path):
    out = io.StringIO()
    guard.check_island_credentials({"AWS_SECRET_ACCESS_KEY": "x", guard.ALLOW_ENV: "1"},
                                   home=tmp_path, stream=out)
    assert "WARNING" in out.getvalue() and "AWS_SECRET_ACCESS_KEY" in out.getvalue()


def test_guard_warns_on_credential_files(tmp_path):
    (tmp_path / ".modal.toml").write_text("[default]\ntoken_secret = 'zzz'\n")
    out = io.StringIO()
    guard.check_island_credentials({}, home=tmp_path, stream=out)
    assert "~/.modal.toml" in out.getvalue() and "zzz" not in out.getvalue()


def test_guard_is_quiet_on_a_clean_island(tmp_path):
    out = io.StringIO()
    guard.check_island_credentials({"HF_TOKEN": "h"}, home=tmp_path, stream=out)
    assert out.getvalue() == ""


@pytest.mark.parametrize("module", ["yeto.learner", "yeto.rl.adapters.verl.island_entry",
                                    "yeto.rl.adapters.miles.island_entry", "yeto.megatron.learner",
                                    "yeto.diffusion.learner"])
def test_island_entry_modules_run_the_guard(module):
    import importlib.util

    source = open(importlib.util.find_spec(module).origin, encoding="utf-8").read()
    main_block = source[source.index('if __name__ == "__main__":'):]
    assert "check_island_credentials()" in main_block
