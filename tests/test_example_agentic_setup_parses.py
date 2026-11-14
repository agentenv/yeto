"""Every shipped spec (examples / tutorial / docs JSON files, and the K3.1 guide's JSON blocks)
that declares ``agentic.setup`` passes the node setup's own reading (``agentic_setup.parse``),
and the guide's tunnel instructions name the records proxy port the tunnel needs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from yeto.launch import agentic_setup

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "KIMI_K31_ASYNC_RL_GUIDE.md"
SPEC_DIRS = ("examples", "tutorial", "docs")


def _filled(doc: Any, tmp: Path) -> Any:
    if isinstance(doc, dict):
        return {k: _filled(v, tmp) for k, v in doc.items() if k != "_comment"}
    if isinstance(doc, list):
        return [_filled(v, tmp) for v in doc]
    if isinstance(doc, str) and "<" in doc:
        return str(tmp / re.sub(r"[<>/ :]", "_", doc))
    return doc


def _has_setup(doc: Any) -> bool:
    return isinstance(doc, dict) and isinstance((doc.get("agentic") or {}).get("setup"), dict)


def _specs() -> list[Any]:
    out = []
    for name in SPEC_DIRS:
        for path in sorted((ROOT / name).rglob("*.json")):
            try:
                doc = json.loads(path.read_text())
            except (ValueError, UnicodeDecodeError):
                continue
            if _has_setup(doc):
                out.append(pytest.param(doc, id=str(path.relative_to(ROOT))))
    text = GUIDE.read_text()
    for m in re.finditer(r"^```json\n(.*?)^```", text, re.S | re.M):
        try:
            doc = json.loads(m.group(1))
        except ValueError:
            continue
        if _has_setup(doc):
            out.append(pytest.param(doc, id=f"guide:{text[:m.start()].count(chr(10)) + 1}"))
    return out


@pytest.mark.parametrize("spec", _specs())
def test_shipped_agentic_setup_parses(spec, tmp_path):
    spec = _filled(spec, tmp_path)
    setup = spec["agentic"]["setup"]
    if isinstance(setup.get("tunnel"), dict) and not re.search(r":\d+(-\d+)?$", str(setup["tunnel"].get("listen", ""))):
        setup["tunnel"]["listen"] = "10.0.0.1:20000-20007"  # a placeholder listen address
    agentic_setup.parse(spec, base_dir=tmp_path)


def test_the_guide_names_the_records_proxy_port_the_tunnel_needs():
    """agentic_setup.parse refuses a tunnel without agentic.proxy.port (or tunnel.target_port)."""
    text = GUIDE.read_text()
    assert "agentic.setup.tunnel" in text
    assert "agentic.proxy.port" in text, "the guide documents agentic.setup.tunnel but not agentic.proxy.port"


def test_a_tunnel_without_a_proxy_port_is_refused(tmp_path):
    key = tmp_path / "key"
    key.write_text("k")
    spec = {"agentic": {"setup": {"sandbox": {"docker_host": "ssh://u@h", "key": {"path": str(key)}},
                                  "tunnel": {"listen": "10.0.0.1:20000"}, "records_secret": "generate"}}}
    with pytest.raises(ValueError, match="fixed records proxy port"):
        agentic_setup.parse(spec, base_dir=tmp_path)
    spec["agentic"]["proxy"] = {"port": 18080}
    assert agentic_setup.parse(spec, base_dir=tmp_path).tunnel.target_port == 18080
