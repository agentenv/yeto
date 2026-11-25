"""Unit tests for :mod:`yeto.rl.harness.harbor.agentenv_network`.

All I/O is mocked: the microVM ``exec`` is a fake async callable returning
scripted ``ExecResult``s, and ``sleep`` is a no-op.  No sandbox, client, pool or
network is touched.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from yeto.rl.harness.harbor import agentenv_network as net
from yeto.rl.harness.harbor.agentenv_network import (
    NetworkError,
    SidecarEndpoint,
    check_reachability,
    egress_allow_cidrs,
    endpoints_from_mapping,
    render_task_text,
    resolve_placeholder_urls,
    self_check,
)


@dataclass
class FakeExec:
    """An async exec_fn returning scripted results, keyed by substring of the command."""

    rules: list[tuple[str, int, str, str]]  # (url_substr, return_code, stdout, stderr)
    calls: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.calls = []

    async def __call__(self, command: str):
        self.calls.append(command)
        for substr, rc, out, err in self.rules:
            if substr in command:
                return _Res(stdout=out, stderr=err, return_code=rc)
        return _Res(stdout="", stderr="no rule", return_code=7)


@dataclass
class _Res:
    stdout: str | None
    stderr: str | None
    return_code: int


async def _noop_sleep(_s: float) -> None:
    return None


# --------------------------------------------------------------------------- #
# endpoint properties
# --------------------------------------------------------------------------- #


def test_endpoint_host_port_private():
    ep = SidecarEndpoint("task-server", "http://10.0.0.5:8300")
    assert ep.host == "10.0.0.5"
    assert ep.port == 8300
    assert ep.private is True


def test_endpoint_scheme_default_port_and_public():
    assert SidecarEndpoint("x", "https://judge.example.com").port == 443
    assert SidecarEndpoint("x", "http://judge.example.com").port == 80
    assert SidecarEndpoint("x", "https://judge.example.com").private is False  # DNS name = public here


def test_endpoint_loopback_and_linklocal_private():
    assert SidecarEndpoint("x", "http://127.0.0.1:9").private is True
    assert SidecarEndpoint("x", "http://169.254.1.1:9").private is True


def test_probe_url_joins_health_path():
    ep = SidecarEndpoint("x", "http://10.0.0.5:8300/", health_path="/health")
    assert ep.probe_url() == "http://10.0.0.5:8300/health"
    assert SidecarEndpoint("x", "http://10.0.0.5:8300").probe_url() == "http://10.0.0.5:8300"


# --------------------------------------------------------------------------- #
# egress allow-list
# --------------------------------------------------------------------------- #


def test_egress_allow_cidrs_only_private_ips():
    eps = [
        SidecarEndpoint("a", "http://10.0.0.5:8300"),
        SidecarEndpoint("b", "http://192.168.1.9:8400"),
        SidecarEndpoint("c", "https://public.example.com"),  # public DNS -> no rule
        SidecarEndpoint("d", "http://8.8.8.8:80"),  # public IP -> no rule
    ]
    assert egress_allow_cidrs(eps) == ["10.0.0.5/32", "192.168.1.9/32"]


def test_egress_allow_cidrs_dedup_and_ipv6():
    eps = [
        SidecarEndpoint("a", "http://10.0.0.5:8300"),
        SidecarEndpoint("a2", "http://10.0.0.5:8301"),  # same host, dedup
        SidecarEndpoint("v6", "http://[fd00::1]:8300"),  # ULA private
    ]
    assert egress_allow_cidrs(eps) == ["10.0.0.5/32", "fd00::1/128"]


# --------------------------------------------------------------------------- #
# placeholder rendering
# --------------------------------------------------------------------------- #


def test_resolve_placeholder_urls():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300")]
    table = resolve_placeholder_urls(eps)
    assert table["ts"] == {"url": "http://10.0.0.5:8300", "host": "10.0.0.5", "port": "8300"}


def test_render_task_text_substitutes_url_host_port():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300")]
    text = "curl {{yeto.sidecar.ts.url}}/verify host={{yeto.sidecar.ts.host}} p={{yeto.sidecar.ts.port}}"
    assert render_task_text(text, eps) == "curl http://10.0.0.5:8300/verify host=10.0.0.5 p=8300"


def test_render_task_text_unknown_sidecar_raises():
    with pytest.raises(NetworkError, match="unknown sidecar 'nope'"):
        render_task_text("{{yeto.sidecar.nope.url}}", [SidecarEndpoint("ts", "http://10.0.0.5:8300")])


def test_render_task_text_bridge_ip_rejected():
    # The remote-docker-only placeholder must fail loud under agentenv.
    with pytest.raises(NetworkError, match="bridge"):
        render_task_text("ip={{yeto.host.bridge_ip}}", [])


# --------------------------------------------------------------------------- #
# endpoints_from_mapping
# --------------------------------------------------------------------------- #


def test_endpoints_from_mapping_strings_and_dicts():
    spec = {
        "task-server": "http://10.0.0.5:8300",
        "records-proxy": {"url": "https://rp.example.com", "health": "/healthz"},
    }
    eps = {e.name: e for e in endpoints_from_mapping(spec, optional=["records-proxy"])}
    assert eps["task-server"].required is True
    assert eps["task-server"].health_path is None
    assert eps["records-proxy"].required is False
    assert eps["records-proxy"].health_path == "/healthz"


def test_endpoints_from_mapping_bad_value():
    with pytest.raises(NetworkError, match="must be a URL string or a mapping"):
        endpoints_from_mapping({"x": 123})
    with pytest.raises(NetworkError, match="no string 'url'"):
        endpoints_from_mapping({"x": {"health": "/h"}})


# --------------------------------------------------------------------------- #
# reachability self-check
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_check_reachability_all_healthy():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300", health_path="/health")]
    exec_fn = FakeExec(rules=[("10.0.0.5", 0, "200 0.012", "")])
    report = await check_reachability(exec_fn, eps, sleep=_noop_sleep)
    assert report.ok is True
    h = report.by_name("ts")
    assert h.reachable and h.healthy
    assert h.http_code == 200 and h.latency_s == pytest.approx(0.012)
    assert len(exec_fn.calls) == 1  # healthy on first try, no retries


@pytest.mark.asyncio
async def test_check_reachability_retries_then_succeeds():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300", health_path="/health")]
    # First two attempts connection-refused (exit 7), third 200.
    calls = {"n": 0}

    async def exec_fn(cmd: str):
        calls["n"] += 1
        if calls["n"] < 3:
            return _Res(stdout="", stderr="refused", return_code=7)
        return _Res(stdout="200 0.1", stderr="", return_code=0)

    report = await check_reachability(exec_fn, eps, tries=5, sleep=_noop_sleep)
    assert report.ok is True
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_check_reachability_unreachable_reported_not_raised():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300", health_path="/health")]
    exec_fn = FakeExec(rules=[("10.0.0.5", 7, "", "Connection refused")])
    report = await check_reachability(exec_fn, eps, tries=3, sleep=_noop_sleep)
    assert report.ok is False
    h = report.by_name("ts")
    assert not h.reachable and not h.healthy
    assert "network unreachable" in h.error
    assert len(exec_fn.calls) == 3  # exhausted retries


@pytest.mark.asyncio
async def test_reached_but_http_error_is_reachable_not_healthy():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300", health_path="/health")]
    exec_fn = FakeExec(rules=[("10.0.0.5", 0, "503 0.05", "")])
    report = await check_reachability(exec_fn, eps, tries=2, sleep=_noop_sleep)
    h = report.by_name("ts")
    assert h.reachable is True and h.healthy is False
    assert h.http_code == 503 and "HTTP 503" in h.error
    assert report.ok is False


@pytest.mark.asyncio
async def test_no_health_path_any_reply_is_healthy():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300")]  # no health_path
    exec_fn = FakeExec(rules=[("10.0.0.5", 0, "404 0.01", "")])
    report = await check_reachability(exec_fn, eps, sleep=_noop_sleep)
    h = report.by_name("ts")
    assert h.reachable and h.healthy  # a reply at all is enough without a health path


@pytest.mark.asyncio
async def test_malformed_url_fails_without_exec():
    eps = [SidecarEndpoint("ts", "ftp://nope")]
    exec_fn = FakeExec(rules=[])
    report = await check_reachability(exec_fn, eps, sleep=_noop_sleep)
    assert report.ok is False
    assert "malformed" in report.by_name("ts").error
    assert exec_fn.calls == []  # never probed


@pytest.mark.asyncio
async def test_optional_endpoint_failure_keeps_ok_true():
    eps = [
        SidecarEndpoint("ts", "http://10.0.0.5:8300", health_path="/health", required=True),
        SidecarEndpoint("rp", "http://10.0.0.6:8400", health_path="/health", required=False),
    ]
    exec_fn = FakeExec(rules=[("10.0.0.5", 0, "200 0.01", ""), ("10.0.0.6", 7, "", "refused")])
    report = await check_reachability(exec_fn, eps, tries=1, sleep=_noop_sleep)
    assert report.ok is True  # only the optional one failed
    assert not report.by_name("rp").healthy


@pytest.mark.asyncio
async def test_self_check_raises_on_required_failure():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300", health_path="/health")]
    exec_fn = FakeExec(rules=[("10.0.0.5", 28, "", "timed out")])
    with pytest.raises(NetworkError, match="required sidecar"):
        await self_check(exec_fn, eps, tries=1, sleep=_noop_sleep)


@pytest.mark.asyncio
async def test_self_check_returns_report_on_success():
    eps = [SidecarEndpoint("ts", "http://10.0.0.5:8300", health_path="/health")]
    exec_fn = FakeExec(rules=[("10.0.0.5", 0, "204 0.02", "")])
    report = await self_check(exec_fn, eps, tries=1, sleep=_noop_sleep)
    assert report.ok is True
    assert report.by_name("ts").http_code == 204


@pytest.mark.asyncio
async def test_tries_zero_rejected():
    with pytest.raises(ValueError, match="tries must be >= 1"):
        await check_reachability(FakeExec(rules=[]), [], tries=0, sleep=_noop_sleep)


# --------------------------------------------------------------------------- #
# probe command shape
# --------------------------------------------------------------------------- #


def test_probe_command_is_body_free_and_bounded():
    cmd = net._probe_command("http://10.0.0.5:8300/health", timeout_s=5)
    assert "-o /dev/null" in cmd
    assert "--max-time 5" in cmd
    assert "http://10.0.0.5:8300/health" in cmd
    assert "%{http_code}" in cmd


# --------------------------------------------------------------------------- #
# probe fallbacks for task images without curl (run the real command in bash)
# --------------------------------------------------------------------------- #


def _serve(code: int):
    import http.server
    import threading

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(code)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_in(tools: list[str], tmp_path, cmd: str) -> _Res:
    """Run the probe in bash with PATH holding only ``tools`` (no curl, maybe no python)."""
    import shutil
    import subprocess

    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    for tool in tools:
        found = shutil.which(tool)
        if found and not (bindir / tool).exists():
            (bindir / tool).symlink_to(found)
    bash = shutil.which("bash")
    proc = subprocess.run([bash, "-c", cmd], capture_output=True, text=True, timeout=60,
                          env={"PATH": str(bindir)})
    return _Res(stdout=proc.stdout, stderr=proc.stderr, return_code=proc.returncode)


def test_probe_command_tries_curl_first_then_python_then_tcp():
    cmd = net._probe_command("http://10.0.0.5:8300/health", timeout_s=5)
    assert cmd.index("curl -sS") < cmd.index('"$p" -c') < cmd.index("/dev/tcp/")


def test_probe_without_curl_uses_python_and_reads_like_curl(tmp_path):
    srv = _serve(204)
    try:
        ep = SidecarEndpoint("rp", f"http://127.0.0.1:{srv.server_port}", health_path="/health")
        res = _run_in(["bash", "python3"], tmp_path, net._probe_command(ep.probe_url(), timeout_s=5))
    finally:
        srv.shutdown()
    assert res.return_code == 0 and res.stdout.split()[0] == "204"
    health = net._interpret(ep, res)
    assert health.reachable and health.healthy and health.http_code == 204


def test_python_probe_reports_a_refused_connection_as_unreachable(tmp_path):
    ep = SidecarEndpoint("rp", f"http://127.0.0.1:{_free_port()}")
    res = _run_in(["bash", "python3"], tmp_path, net._probe_command(ep.probe_url(), timeout_s=5))
    assert res.return_code == 7
    health = net._interpret(ep, res)
    assert not health.reachable and "network unreachable" in health.error


def test_probe_without_curl_or_python_connects_by_tcp(tmp_path):
    srv = _serve(200)
    tools = ["bash", "date", "awk", "timeout"]
    try:
        bare = SidecarEndpoint("rp", f"http://127.0.0.1:{srv.server_port}")
        res = _run_in(tools, tmp_path, net._probe_command(bare.probe_url(), timeout_s=5))
        checked = SidecarEndpoint("ts", f"http://127.0.0.1:{srv.server_port}", health_path="/health")
        res_checked = _run_in(tools, tmp_path, net._probe_command(checked.probe_url(), timeout_s=5))
    finally:
        srv.shutdown()
    assert res.return_code == 0 and res.stdout.split()[0] == net.TCP_ONLY
    health = net._interpret(bare, res)
    assert health.reachable and health.healthy and health.http_code is None
    # a health path cannot be checked without an HTTP client: reached, not healthy, and said so
    health = net._interpret(checked, res_checked)
    assert health.reachable and not health.healthy and "TCP only" in health.error


def test_tcp_probe_reports_a_refused_connection_as_unreachable(tmp_path):
    ep = SidecarEndpoint("rp", f"http://127.0.0.1:{_free_port()}")
    res = _run_in(["bash", "date", "awk", "timeout"], tmp_path, net._probe_command(ep.probe_url(), timeout_s=5))
    assert res.return_code == 7 and "TCP connect" in res.stderr
    health = net._interpret(ep, res)
    assert not health.reachable and "network unreachable" in health.error


@pytest.mark.asyncio
async def test_curl_000_on_a_failed_connection_is_unreachable():
    # curl -w '%{http_code}' prints 000 when no HTTP reply came: that is no reply, not a reached service
    eps = [SidecarEndpoint("rp", "http://10.0.0.5:8400")]
    exec_fn = FakeExec(rules=[("10.0.0.5", 7, "000 0.001589", "Failed to connect")])
    report = await check_reachability(exec_fn, eps, tries=1, sleep=_noop_sleep)
    health = report.by_name("rp")
    assert not health.reachable and not health.healthy and health.http_code is None
    assert "network unreachable" in health.error and report.ok is False
