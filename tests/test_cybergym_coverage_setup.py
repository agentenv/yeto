import importlib.util
import sys
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parent.parent / "scripts" / "prepare_cybergym_coverage.py"
    spec = importlib.util.spec_from_file_location("prepare_cybergym_coverage", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


coverage = _load()


def test_arvo_wrapper_uses_afl_showmap_file_mode(tmp_path):
    runner = tmp_path / "runner"
    runner.write_text("export ASAN_OPTIONS=detect_leaks=0\n/out/target /tmp/poc\n")

    wrapper = coverage.arvo_wrapper(runner)

    assert "/src/aflplusplus/afl-showmap" in wrapper
    assert "-- env LD_LIBRARY_PATH=/out-libs /out/target /tmp/poc" in wrapper
    assert "-print_coverage=1" not in wrapper
    assert "coverage-map.json" in wrapper
