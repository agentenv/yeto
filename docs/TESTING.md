# Testing

"Done" for a code change means: the local safe suite below passes (no
failures, no errors; every skip has a reason) and CI is green.

```bash
python -m pytest tests      # local safe suite
cargo test --manifest-path syncer/Cargo.toml
```

## 1. Create a test venv (once)

Use a dedicated venv. Do not install into a shared runtime venv.

```bash
cd /path/to/yeto
uv venv -p 3.12 <test-venv>
uv pip install -p <test-venv>/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch==2.8.0+cpu" -r pyproject.toml --extra test
```

- The `test` extra in `pyproject.toml` pins skypilot, boto3, peft, accelerate
  and pylatexenc, and adds pytest, pytest-timeout, pytest-asyncio and pillow.
- The venv does not install ray, so no unit test can start Ray there.
- On macOS, drop the CPU index and `+cpu`: `"torch==2.8.0"`.
- To build the Rust syncer for the syncer tests, install cargo and put
  `~/.cargo/bin` on `PATH`. Without cargo, those tests skip with a reason.

## 2. Run the local safe suite

Make sure no GPU launch chain runs on the same machine first (thread budget).

```bash
cd /path/to/yeto && PATH="$HOME/.cargo/bin:$PATH" PYTHONPATH=. \
  <test-venv>/bin/python -m pytest tests -q -p no:cacheprovider -rfEs
```

## 3. Tests that start Ray (`ray_local`)

Tests that start Ray on the local machine have the `ray_local` marker.

- By default, pytest does not collect them (they show as "deselected").
- To run them, add `--run-ray-local`. Run them only inside the training
  image or on a machine with no GPU launch chain.
- In a default run, a test without the marker that calls `ray.init` (also
  Ray's auto-init) fails with a message that tells you to add the marker.
  The guard cannot stop a subprocess that runs `ray start`.

## 4. Skips

A skip is allowed only for a missing environment item (an optional package,
cargo, CUDA, an external checkout, a syncer binary via `YETO_TEST_SYNCER`). The
reason must say what is missing and where the test runs. Do not skip a test
when the test and the code disagree: fix the side that is wrong.

Tests that need Hugging Face tokenizer/config files take them from the local
HF cache through `tests/hf_assets.py` (`require_hf_repo`), which pins every repo
to a revision. Fetch them once (about 110 MB, no weights) with
`python scripts/prefetch_hf_test_assets.py`; without them these tests skip.
CI caches the HF hub directory keyed by the repo list, runs with
`HF_HUB_OFFLINE=1`, and sets `YETO_REQUIRE_HF_ASSETS=1` so a missing repo fails.

## 5. CI

CI runs on every pull request and every push to `main`:

- **rust**: the syncer test suite (`cargo test --manifest-path syncer/Cargo.toml`).
- **python** (CPU): a repository hygiene check (`scripts/precommit_check.py`:
  file size, secrets, e-mail addresses, private IPs, home paths), then the
  Python suite, which includes a live syncer + learner loop.

## 6. Pre-commit hooks

Install the same local checks once per clone (commit messages must be English):

```bash
pre-commit install && pre-commit install --hook-type commit-msg
```

To run the hygiene check over the whole tree:

```bash
git ls-files -z | xargs -0 python3 scripts/precommit_check.py
```

## 7. Benchmark harnesses

Heavier harnesses need real hardware; all accept `--dry-run` to check the
arguments and print the plan first:

```bash
python scripts/smoke_models.py --tier small --dry-run
python scripts/compare_diloco.py --data <chat.jsonl> --settings all --dry-run
python scripts/benchmark_rl.py <model/data/reward arguments> --dry-run
python scripts/benchmark_diffusion_diloco.py --model Lightricks/LTX-Video \
    --data <media-dataset> --height 512 --width 512 --settings all --dry-run
```

See [RL_BENCHMARK.md](RL_BENCHMARK.md), [LM_BENCHMARK.md](LM_BENCHMARK.md) and
[DIFFUSION_BENCHMARK.md](DIFFUSION_BENCHMARK.md).
