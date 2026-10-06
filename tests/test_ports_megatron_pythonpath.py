"""S11 try25 fnA: megatron.post_training unresolvable in the ports image.

The image's Megatron-LM is a setuptools editable install whose finder maps
only megatron.core and megatron.training ('megatron' itself is an empty
namespace).  This rebuilds that finder from setuptools' own template, with
the exact MAPPING/NAMESPACES recorded in the image layer, and shows that the
island PYTHONPATH the launcher now emits makes megatron.post_training import.
"""

import subprocess
import sys
import textwrap

import pytest

from yeto.launcher import PORTS_MEGATRON_PATH

editable_wheel = pytest.importorskip("setuptools.command.editable_wheel")


def _image_like(tmp_path):
    root = tmp_path / "Megatron-LM"
    for pkg in ("core", "training", "post_training"):
        d = root / "megatron" / pkg
        d.mkdir(parents=True)
        (d / "__init__.py").write_text("")
    (root / "megatron" / "post_training" / "checkpointing.py").write_text(
        "def has_modelopt_state(*a, **k):\n    return False\n"
    )
    site = tmp_path / "site"
    site.mkdir()
    name = "__editable___megatron_core_finder"
    mapping = {
        "megatron.core": str(root / "megatron" / "core"),
        "megatron.training": str(root / "megatron" / "training"),
    }
    (site / f"{name}.py").write_text(editable_wheel._finder_template(name, mapping, {"megatron": []}))
    return root, site, name


def _import(site, name, pythonpath):
    code = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(site)!r})
        import {name}; {name}.install()
        import megatron.training
        from megatron.post_training.checkpointing import has_modelopt_state
        """
    )
    env = {"PYTHONPATH": pythonpath, "PATH": "/usr/bin:/bin"}
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd="/")


def test_editable_finder_alone_cannot_import_post_training(tmp_path):
    _root, site, name = _image_like(tmp_path)
    r = _import(site, name, "")
    assert r.returncode != 0
    assert "No module named 'megatron.post_training'" in r.stderr


def test_checkout_on_pythonpath_imports_post_training(tmp_path):
    root, site, name = _image_like(tmp_path)
    r = _import(site, name, str(root))
    assert r.returncode == 0, r.stderr


def test_ports_megatron_path_is_the_image_checkout():
    assert PORTS_MEGATRON_PATH == "/root/Megatron-LM"
