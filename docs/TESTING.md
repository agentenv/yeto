# Local test suite

"Done" for a code change means: the command below passes on your machine
(no failures, no errors; every skip has a reason). The GitHub CI result is
not used (PM decision, 2026-10-09).

## 1. Create the test venv (once)

Use a dedicated venv. Do not install into a shared runtime venv.

```bash
cd <repo root>
uv venv -p 3.12 /home/michael/work/yeto-test-venv
uv pip install -p /home/michael/work/yeto-test-venv/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.8.0+cpu" -r pyproject.toml --extra test
```

- The `test` extra in `pyproject.toml` pins skypilot, boto3, peft, accelerate
  and pylatexenc, and adds pytest, pytest-asyncio and pillow.
- The venv does not install ray. Then no unit test can start Ray here.
- On macOS, drop the CPU index and `+cpu`: `"torch==2.8.0"`.
- To build the Rust syncer for the syncer tests, install cargo and put
  `~/.cargo/bin` on `PATH`. Without cargo, those tests skip with a reason.

## 2. Run the local safe suite

Make sure no GPU launch chain runs on this machine first (thread budget).

```bash
cd <repo root> && PATH="$HOME/.cargo/bin:$PATH" PYTHONPATH=. \
  /home/michael/work/yeto-test-venv/bin/python -m pytest tests -q -p no:cacheprovider -m "not gpu" -rfEs
```

Last result (2026-10-09, fix-known-red-tests): see the table at the end.

## 3. Tests that start Ray (`ray_local`)

Tests that start Ray on this machine have the `ray_local` marker.

- By default, pytest does not collect them (they show as "deselected").
- To run them, add `--run-ray-local`. Run them only inside the training
  image or on a machine with no GPU launch chain. Never run them here.
- In a default run, a test without the marker that calls `ray.init` (also
  Ray's auto-init) fails with a message that tells you to add the marker.
  The guard cannot stop a subprocess that runs `ray start`.

## 4. Skips

A skip is allowed only for a missing environment item (an optional package,
cargo, CUDA, an external checkout). The reason must say what is missing and
where the test runs. Do not skip a test when the test and the code disagree:
fix the side that is wrong.

## Last result

| Date | Commit | Passed | Skipped | Deselected (ray_local) | Failed | Time |
|---|---|---|---|---|---|---|
| 2026-10-09 | s18-fkrt (base fc82f175) | 5055 | 45 | 23 | 0 | 413 s |

The 45 skips name the missing item: codex-cli 0.145.0, an elastic syncer
binary (`YETO_TEST_ELASTIC_SYNCER`), external adapters (yeto_miles_secrlenv,
SecRLEnv, CyberGym, swebench), upstream Miles/Megatron/mlx, CUDA, ray (the
guard self-tests; they pass in a venv with ray), and a trimmed evidence log.
