"""Registry engine images on Modal (yeto.cloud.modal_images): the vLLM images have
``python3`` but no ``python``, which Modal needs to read the interpreter version.
The image builds of the Modal rollout provider and the Modal engine sampler go
through the helper. A fake ``modal`` records the build; no Modal call is made."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from yeto.cloud import modal_engine_sampler as mes
from yeto.cloud.modal_images import PYTHON_ON_PATH, engine_registry_image
from yeto.launch.pool.modal_provider import ModalSdkClient

VLLM_IMAGE = "vllm/vllm-openai@sha256:" + "8" * 64


class _Image:
    def __init__(self, log, step):
        self.log = log
        log.append(step)

    def __getattr__(self, name):
        return lambda *a, **kw: _Image(self.log, (name, a, kw))


class FakeModal:
    """Just enough of ``modal`` for ``build_app``: records every image step."""

    def __init__(self):
        self.steps, self.functions = [], []
        outer = self

        class Image:
            @staticmethod
            def from_registry(ref, **kw):
                return _Image(outer.steps, ("from_registry", (ref,), kw))

            @staticmethod
            def debian_slim(**kw):  # not used by an engine image
                raise AssertionError("an engine image is built from the registry")

        class App:
            def __init__(self, name):
                self.name = name

            def function(self, **kw):
                outer.functions.append(kw)
                return lambda fn: fn

        handle = SimpleNamespace(read_only=lambda: "ro-volume")
        self.Image, self.App = Image, App
        self.Volume = SimpleNamespace(from_name=lambda *a, **kw: handle)
        self.Secret = SimpleNamespace(from_name=lambda name: f"secret:{name}")


def test_the_helper_links_python3_as_python_without_add_python():
    modal = FakeModal()
    engine_registry_image(modal, VLLM_IMAGE)
    name, args, kw = modal.steps[0]
    assert (name, args) == ("from_registry", (VLLM_IMAGE,))
    # add_python would put a standalone interpreter without the engine's packages ahead on PATH
    assert kw == {}
    # one layer on top of the registry layer (a pulled registry image is reused)
    assert modal.steps[1] == ("dockerfile_commands", ([PYTHON_ON_PATH],), {})


def _run_guard(tmp_path, *, have_python: bool, have_python3: bool = True) -> str:
    """Run the guard's shell (``/usr/local/bin`` redirected to a temp dir) with a PATH
    holding only fake interpreters; returns where ``python`` resolves afterwards."""
    bin_dir, local_bin = tmp_path / "usr-bin", tmp_path / "usr-local-bin"
    bin_dir.mkdir(parents=True)
    local_bin.mkdir()  # /usr/local/bin exists in every image
    for name, present in (("python3", have_python3), ("python", have_python)):
        if present:
            exe = bin_dir / name
            exe.write_text(f"#!/bin/sh\necho {name}\n")
            exe.chmod(0o755)
    tools = tmp_path / "tools"  # only the commands the guard runs: no host python on PATH
    tools.mkdir()
    for tool in ("ln",):
        (tools / tool).symlink_to(shutil.which(tool))
    script = PYTHON_ON_PATH.removeprefix("RUN ").replace("/usr/local/bin", str(local_bin))
    env = {"PATH": f"{local_bin}:{bin_dir}:{tools}"}
    subprocess.run(["/bin/sh", "-c", script], check=True, env=env)
    resolved = subprocess.run(["/bin/sh", "-c", "command -v python || true"], env=env,
                              capture_output=True, text=True).stdout.strip()
    return resolved


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_the_guard_links_python_only_when_the_image_lacks_it(tmp_path):
    # a vLLM image: python3 only -> python is a link to the image's own python3
    linked = _run_guard(tmp_path / "vllm", have_python=False)
    assert linked.endswith("usr-local-bin/python")
    assert os.path.realpath(linked) == str((tmp_path / "vllm" / "usr-bin" / "python3").resolve())


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell")
def test_the_guard_keeps_an_image_python_and_tolerates_none(tmp_path):
    # an image with its own python (the SGLang image) is left as it is
    kept = _run_guard(tmp_path / "sglang", have_python=True)
    assert kept == str(tmp_path / "sglang" / "usr-bin" / "python")
    assert not (tmp_path / "sglang" / "usr-local-bin" / "python").exists()
    # an image with neither (add_python's case) still builds; Modal reports it as before
    assert _run_guard(tmp_path / "bare", have_python=False, have_python3=False) == ""


def test_the_rollout_provider_builds_its_image_through_the_helper():
    modal = FakeModal()
    client = ModalSdkClient({"image": VLLM_IMAGE, "secret": "s", "gpu": "L40S", "gpus": 1,
                             "model_volume": "models"}, run_id="r", modal=modal)
    client.build_app()
    assert modal.steps[:2] == [("from_registry", (VLLM_IMAGE,), {}), ("dockerfile_commands", ([PYTHON_ON_PATH],), {})]
    # the rest of the image is unchanged: entrypoint off, env, yeto source mounted
    assert [s[0] for s in modal.steps[2:]] == ["entrypoint", "env", "add_local_python_source"]
    assert modal.steps[2][1] == ([],)
    assert modal.functions[0]["gpu"] == "L40S:1"


def test_the_engine_sampler_builds_its_image_through_the_helper(tmp_path, monkeypatch):
    modal = FakeModal()
    monkeypatch.setitem(sys.modules, "modal", modal)
    prompts = tmp_path / "p.jsonl"
    prompts.write_text("{}\n")
    c = mes.SamplerConfig(run="r", shard="s0", bucket="b", selection_sha256="ab" * 32, prompts=str(prompts),
                          image=VLLM_IMAGE, secret="store")
    mes.build_app(c)
    assert modal.steps[:2] == [("from_registry", (VLLM_IMAGE,), {}), ("dockerfile_commands", ([PYTHON_ON_PATH],), {})]
    assert [s[0] for s in modal.steps[2:]] == ["entrypoint", "env", "add_local_file", "add_local_python_source"]
