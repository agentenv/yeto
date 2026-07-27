import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/launch_v9_stage.py"


def load():
    spec = importlib.util.spec_from_file_location("launch_v9_stage", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_remote_queue_argv_is_exact_for_one_and_four_gpu_slots():
    module = load()
    for stage, slot in (
        ("stage_1p7b", "h200-n1-gpu3"),
        ("stage_7b", "h200-n2-gpu4-7"),
    ):
        argv = module.remote_queue_argv(
            node="h200-n1" if "n1" in slot else "h200-n2",
            stage=stage,
            slot_id=slot,
            remote_root="/root/control",
            attempt=1,
            retry_name=None,
        )
        assert argv[0] == "/root/yeto-venv/bin/python"
        assert argv[1] == "/root/yeto/scripts/run_slot_v9.py"
        assert argv[argv.index("--stage") + 1] == stage
        assert argv[argv.index("--slot-id") + 1] == slot
        assert "--retry-authority" not in argv


def test_retry_argv_binds_registered_retry_authority_name():
    module = load()
    argv = module.remote_queue_argv(
        node="h200-n1",
        stage="stage_1p7b",
        slot_id="h200-n1-gpu0",
        remote_root="/root/control",
        attempt=2,
        retry_name="retry.json",
    )
    assert argv[argv.index("--attempt") + 1] == "2"
    assert argv[argv.index("--retry-authority") + 1] == "/root/control/retry.json"
