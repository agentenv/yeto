"""Start a trainer from a policy release of its own store (``--rl-init-from-release``).

A trainer relaunched over a store that holds releases continues the *numbering* at
the highest release vK (``merged_release.ResumedReleases``), but without a round
cut its weights are the base checkpoint's. A round cut of a full-parameter
Kimi-K3.1 trainer holds the bf16 copy, the fp32 master and both Adam moments of
every trained expert (about 5 TB), so runs go without one. This entry makes the
cheap restart: the trainer loads release vK's weights instead of the base and
starts its optimizer fresh, so the numbering it continues at names the weights it
holds.

The release is the store's highest version (a lower one is refused: vK would then
name weights the trainer does not hold, and its next release would follow weights
other than vK). Every node of the island runs :func:`prepare` before its Ray
starts (``python -m yeto.rl.engine.init_from_release``): it downloads the release
with the store's object API (``PolicyReleaseStore.resolve``: the files in
parallel, sizes checked) and lays it over the base checkpoint
(``release_overlay.overlay_release`` into :data:`MODEL_SUBDIR` for a partial-experts
release; a full release is used as it is) and writes :data:`RECORD` (the release
record) beside it; the island entry takes that model directory (:func:`island_model_path`) and the
release wiring checks that the resumed numbering starts at that very release
(:func:`check_resumed`).

Not restored: the optimizer state (fp32 master rebuilt from the loaded bf16
weights, Adam moments and step from zero), the LR schedule position and the RNG.
The reference model (``--ref-load``) is the release as well.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .policy_release import PolicyRelease, PolicyReleaseStore
from .release_overlay import is_partial_release, overlay_release

RECORD = "INIT_RELEASE.json"
MODEL_SUBDIR = "model"  # the overlay, beside the record
LATEST_SELECTOR = "latest"
_NAME = re.compile(r"r(\d+)-v(\d+)")


class InitFromReleaseError(ValueError):
    pass


def select_release(records: Iterable[PolicyRelease], selector: str) -> PolicyRelease:
    """The release to start from: the trained release (not derived) with the highest
    version, of the highest seq among those. ``selector``: ``latest``, or that
    release's ``r<seq>-v<version>`` / version number (checked against it)."""
    trained = [r for r in records if r.derived_from is None and r.policy_hash]
    if not trained:
        raise InitFromReleaseError("the store holds no trained release to start from")
    top = max(trained, key=lambda r: (int(r.version), int(r.seq)))
    selector = str(selector).strip()
    if selector == LATEST_SELECTOR:
        return top
    match = _NAME.fullmatch(selector)
    if match:
        wanted = next((r for r in trained if (int(r.seq), int(r.version)) == tuple(map(int, match.groups()))), None)
    elif selector.isdigit():
        wanted = max((r for r in trained if int(r.version) == int(selector)), key=lambda r: int(r.seq), default=None)
    else:
        raise InitFromReleaseError(f"release selector {selector!r}: 'latest', 'r<seq>-v<version>' or a version")
    if wanted is None:
        raise InitFromReleaseError(f"the store holds no release {selector}")
    if wanted.path != top.path:
        raise InitFromReleaseError(
            f"release {wanted.path} is not the store's highest version ({top.path}): the trainer "
            f"continues its numbering at v{top.version}, which would then name weights it does not hold")
    return wanted


def check_selection(release: PolicyRelease, expected_selection_sha256: str | None) -> None:
    """A partial-experts release must carry the running expert selection."""
    if not is_partial_release(release.manifest):
        return
    actual = (release.manifest or {}).get("expert_selection_sha256")
    if expected_selection_sha256 is None:
        raise InitFromReleaseError(f"{release.path} is a partial-experts release: the trainer needs "
                                   "--rl-expert-selection")
    if actual != expected_selection_sha256:
        raise InitFromReleaseError(f"{release.path} was trained with expert selection {actual}, "
                                   f"the trainer runs {expected_selection_sha256}")


def prepare(releases: PolicyReleaseStore, selector: str, base_dir: str | os.PathLike[str],
            out_dir: str | os.PathLike[str], *, expected_selection_sha256: str | None = None) -> dict[str, Any]:
    """Download the selected release and make ``out_dir`` the trainer's model
    directory (module doc). Returns the record written to ``out_dir/RECORD``."""
    release = select_release(releases.records(), selector)
    check_selection(release, expected_selection_sha256)
    weights = releases.resolve(release)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if is_partial_release(release.manifest):
        # the record stays outside the model directory: the trainer's exports copy
        # its non-weight files into every release
        directory = overlay_release(base_dir, weights, out / MODEL_SUBDIR)
    else:
        directory = Path(weights)
    record = {"path": release.path, "seq": int(release.seq), "version": int(release.version),
              "policy_hash": release.policy_hash, "model_dir": str(directory),
              "base_dir": str(Path(base_dir)), "partial": is_partial_release(release.manifest),
              "expert_selection_sha256": (release.manifest or {}).get("expert_selection_sha256")}
    (out / RECORD).write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def read_record(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(out_dir).expanduser() / RECORD
    if not path.is_file():
        raise InitFromReleaseError(f"{path} is missing: the node did not prepare the release")
    return json.loads(path.read_text())


def island_model_path(out_dir: str | os.PathLike[str], base_path: str | os.PathLike[str]) -> tuple[str, dict]:
    """The model directory the island loads instead of ``base_path`` and the record;
    raises when the release was laid over another base."""
    record = read_record(out_dir)
    if record["partial"] and Path(record["base_dir"]).resolve() != Path(base_path).resolve():
        raise InitFromReleaseError(f"{record['path']} was laid over {record['base_dir']}, "
                                   f"the island's base is {base_path}")
    return str(record["model_dir"]), record


def check_resumed(record: Mapping[str, Any], base_version: int | None, history: Iterable[tuple[int, str]]) -> None:
    """The resumed numbering (``ResumedReleases``) must start at the loaded release:
    same version, and the store names that version with the release's policy hash."""
    if base_version is None or int(base_version) != int(record["version"]):
        raise InitFromReleaseError(f"the trainer loaded {record['path']} but resumes its numbering at "
                                   f"v{base_version}")
    hashes = dict(history)
    if hashes.get(int(record["version"])) != record["policy_hash"]:
        raise InitFromReleaseError(f"the store names v{record['version']} with policy hash "
                                   f"{hashes.get(int(record['version']))}, the loaded release has "
                                   f"{record['policy_hash']}")


def _base_path(model: str, revision: str | None) -> str:
    if Path(model).expanduser().is_dir():
        return str(Path(model).expanduser().resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model, revision=revision)


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--release", required=True, help="latest, r<seq>-v<version> or a version")
    parser.add_argument("--store", default=None, help="the island's store path (default: the object stage)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--expert-selection", default=None, help="the trainer's selection file")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    env = os.environ if environ is None else environ
    from .resume import store_for

    store_path = args.store or env.get("YETO_RL_STORE_S3_STAGE")
    if not store_path:
        raise InitFromReleaseError("no store path: pass --store or set YETO_RL_STORE_S3_STAGE")
    selection = None
    if args.expert_selection:
        from . import backends  # the selection file format is the backend's (never imported statically)

        selection_file = backends.module("expert_selection").ExpertSelection.load_file
        selection = selection_file(Path(args.expert_selection).expanduser()).sha256
    out = Path(args.out).expanduser()
    releases = PolicyReleaseStore(store_for(store_path, environ=env), cache_dir=out / "files")
    record = prepare(releases, args.release, _base_path(args.model, args.model_revision), out,
                     expected_selection_sha256=selection)
    print(f"[yeto-init-release] {json.dumps(record, sort_keys=True)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
