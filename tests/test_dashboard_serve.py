"""fleet-dashboard 3.5/3.6: loopback GET-only server and the offline export."""

import json
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from yeto.dashboard import cli as dash_cli
from yeto.dashboard.export import export_html, export_view, render_page
from yeto.dashboard.reducer import Reducer
from yeto.dashboard.serve import DashboardState, check_loopback, make_server

from dashboard_helpers import FX, T0, four_island_reducer

PATHS = [str(FX / "s9-m4x1"), str(FX / "kill44" / "syncer.jsonl")]


@pytest.fixture
def server():
    state = DashboardState(Reducer(run="fx"), PATHS)
    state.poll()
    srv = make_server(state, "127.0.0.1", 0, live=False)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield base, state
    srv.shutdown()
    srv.server_close()


def get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.headers.get("Content-Type"), resp.read()


def test_endpoints_return_json(server):
    base, state = server
    _, ctype, body = get(base + "/api/overview")
    ov = json.loads(body)
    assert ctype.startswith("application/json")
    assert {"run", "alerts", "cost", "islands", "series", "metrics", "global_status"} <= set(ov)
    assert ov["islands"][0]["id"] == "0"
    isl = json.loads(get(base + "/api/islands/0")[2])
    assert {"card", "series", "cells", "transactions", "recent_events", "ray_embed"} <= set(isl)
    rounds = json.loads(get(base + "/api/rounds")[2])
    assert rounds["derived"] and len(rounds["rounds"]) == 4
    bad = json.loads(get(base + "/api/rounds?only_bad=1")[2])["rounds"]
    assert [x["round"] for x in bad] == [x["round"] for x in rounds["rounds"] if x["missed"]] != []
    fleet = json.loads(get(base + "/api/fleet")[2])
    assert {"records", "cost_ticks", "cost"} <= set(fleet)
    ev = json.loads(get(base + "/api/events?island=0&type=rl_local_round")[2])
    assert [e["type"] for e in ev["events"]] == ["rl_local_round"] * 2
    status, ctype, page = get(base + "/")
    assert status == 200 and ctype.startswith("text/html") and b'id="yeto-data"' not in page
    with pytest.raises(urllib.error.HTTPError) as e:
        get(base + "/api/islands/nope")
    assert e.value.code == 404


@pytest.mark.parametrize("method", ["POST", "PUT", "DELETE", "PATCH"])
def test_non_get_is_405_and_changes_nothing(server, method):
    base, state = server
    before = state.reducer.event_seq
    req = urllib.request.Request(base + "/api/overview", data=b"{}", method=method)
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 405 and e.value.headers["Allow"] == "GET"
    assert state.reducer.event_seq == before


@pytest.mark.parametrize("host", ["0.0.0.0", "10.0.0.5", "::", "example.com"])
def test_non_loopback_host_is_refused(host):
    with pytest.raises(ValueError, match="loopback"):
        check_loopback(host)
    state = DashboardState(Reducer(), [])
    with pytest.raises(ValueError, match="SSH tunnel"):
        make_server(state, host, 0)


def test_cli_refuses_non_loopback(capsys):
    from yeto.cli import main

    assert main(["dashboard", "serve", "--host", "0.0.0.0", "--tapes", PATHS[0]]) == 2
    assert "loopback" in capsys.readouterr().err


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
def test_loopback_hosts_are_accepted(host):
    check_loopback(host)


def test_export_is_self_contained_and_matches_the_api(server, tmp_path):
    base, state = server
    out = tmp_path / "run.html"
    r = Reducer(run="fx")
    from yeto.dashboard.sources import load_all

    load_all(r, PATHS)
    export_html(r, out, generated_at=T0)
    html = out.read_text()
    assert not re.search(r"https?://", html)
    assert "<link" not in html and "src=\"http" not in html and "@import" not in html
    data = json.loads(re.search(r'<script type="application/json" id="yeto-data">(.*?)</script>',
                                html, re.S).group(1))
    api = json.loads(get(base + "/api/overview")[2])
    assert data["overview"] == api  # same reducer, same (data-derived) clock
    assert data["generated_at"] == T0 and data["overview"]["mode"] == "offline"


def test_export_escapes_script_breakers_and_urls():
    r = Reducer()
    r.feed({"event": "rl_reconfiguration", "island_id": 0, "time_unix": T0, "result": "FAILED",
            "error": "</script><script>alert(1)</script> see https://example.com <!-- x"})
    html = render_page(export_view(r, generated_at=T0))
    block = re.search(r'id="yeto-data">(.*?)</script>', html, re.S).group(1)
    assert "</script" not in block and "<!--" not in block and "://" not in block
    data = json.loads(block)
    assert data["events"]["events"][0]["record"]["error"].startswith("</script>")


def test_cli_export(tmp_path, capsys):
    from yeto.cli import main

    out = tmp_path / "x.html"
    assert main(["dashboard", "export", "--tapes", *PATHS, "-o", str(out), "--budget", "100"]) == 0
    assert out.exists() and "51 record(s)" in capsys.readouterr().out


def test_old_tape_renders_no_data_in_the_page(tmp_path):
    """Run the page script under node with a DOM stub (skipped without node)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    r = four_island_reducer()
    out = export_html(r, tmp_path / "t.html", generated_at=T0)
    res = subprocess.run([node, str(FX.parent.parent / "js" / "dashboard_smoke.js"), str(out)],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    page = json.loads(res.stdout)
    assert "无数据" in page["cards"]  # no heartbeat / resource samples on these tapes
    assert "无数据" in page["eff"]  # no fleet.jsonl
    assert "<svg" in page["chart"] and "<polyline" in page["chart"]
    assert "岛 3" in page["rounds"] and "3/4" in page["rounds"]
    assert "岛明细" in page["drill"] and "待 infra 就绪" in page["drill"]
    assert "grad_norm" in page["chart_grad"]
