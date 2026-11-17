"""tests/hard_exit.py: only the outermost pytest session of a process ends it."""

from __future__ import annotations

import shutil
from pathlib import Path

pytest_plugins = ["pytester"]

HERE = Path(__file__).resolve().parent

# The same wiring as tests/conftest.py, without the rest of that conftest.
WIRING = '''
from hard_exit import note_configure, note_exitstatus, pytest_unconfigure  # noqa: F401


def pytest_configure(config):
    note_configure(config)


def pytest_sessionfinish(session, exitstatus):
    note_exitstatus(session, exitstatus)
'''

FAILING = "def test_fails():\n    assert False\n"


def test_a_nested_in_process_run_returns_to_its_test(pytester):
    # The process's own session is the outer one: a nested run's end must not
    # os._exit with the nested status (the outer run would stop, status 0).
    shutil.copy(HERE / "conftest.py", pytester.path / "conftest.py")
    pytester.makepyfile(test_nested=FAILING)
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    assert result.ret == 1  # reached: the process is still running this test


def test_the_outermost_session_exits_with_its_own_status(pytester):
    shutil.copy(HERE / "hard_exit.py", pytester.path / "hard_exit.py")
    pytester.makeconftest(WIRING)
    pytester.makepyfile(test_outer=FAILING)
    assert pytester.runpytest_subprocess("-p", "no:cacheprovider").ret == 1
    pytester.makepyfile(test_outer="def test_passes():\n    pass\n")
    assert pytester.runpytest_subprocess("-p", "no:cacheprovider").ret == 0
