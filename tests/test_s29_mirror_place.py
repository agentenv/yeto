"""D24: cut120_mirror.sh place must create the download dir before `modal volume get`.

modal 1.2.6 `volume get <remote dir> <missing local path>` writes every file onto that single
path, so the VM saw one 215 MB blob and the manifest check failed (missing RELEASE.json, ...).
"""
import re
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/probes/s29_spot/cut120_mirror.sh"


def _place_body() -> str:
    s = SCRIPT.read_text()
    return s[s.index("\nplace() {"):s.index("\ncase ")]


def test_place_creates_download_dir_before_modal_get():
    body = _place_body()
    get = body.index("modal volume get")
    mk = re.search(r"mkdir -p /root/staged-\$rel-dl", body)
    assert mk and mk.start() < get


def test_place_clears_stale_download_dir():
    body = _place_body()
    rm = body.index("rm -rf /root/staged-$rel /root/staged-$rel-dl")
    assert rm < body.index("mkdir -p /root/staged-$rel-dl")
