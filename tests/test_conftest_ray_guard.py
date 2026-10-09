"""fix-known-red-tests 1.3: ray_local marker, --run-ray-local switch, ray.init guard."""

import shutil
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = Path(__file__).with_name("conftest.py")

SAMPLE = '''
import pytest

@pytest.mark.ray_local
def test_marked():
    pass

def test_plain():
    pass
'''


@pytest.fixture
def suite(pytester):
    shutil.copy(CONFTEST, pytester.path / "conftest.py")
    pytester.makepyfile(test_sample=SAMPLE)
    return pytester


def test_default_deselects_ray_local(suite):
    result = suite.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, deselected=1)


def test_switch_collects_ray_local(suite):
    result = suite.runpytest("-p", "no:cacheprovider", "--run-ray-local")
    result.assert_outcomes(passed=2)


def test_unmarked_ray_init_fails_with_hint(pytester):
    pytest.importorskip("ray")
    shutil.copy(CONFTEST, pytester.path / "conftest.py")
    pytester.makepyfile(test_init='''
import ray

def test_calls_init():
    ray.init()
''')
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*ray_local*"])


def test_guard_does_not_import_ray_and_catches_a_late_import(pytester):
    pytest.importorskip("ray")
    shutil.copy(CONFTEST, pytester.path / "conftest.py")
    pytester.makepyfile(test_late='''
import sys

def test_a_guard_leaves_ray_unimported():
    assert "ray" not in sys.modules

def test_b_import_inside_test_then_init():
    import ray
    ray.init()
''')
    result = pytester.runpytest_subprocess("-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*ray_local*"])
