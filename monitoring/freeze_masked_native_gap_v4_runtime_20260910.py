"""Create the exact self-contained v4 runtime archive consumed by the stager."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import uuid

from monitoring.stage_masked_native_gap_v4_20260910 import REQUIRED, SCHEMA, _safe, sha


def _tar_info(name, size):
    info = tarfile.TarInfo(name)
    info.size = size
    info.mode = 0o444
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def _assert_source_stable(root, signatures):
    for name, expected in signatures.items():
        value = (root / name).stat()
        if expected != (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                        value.st_ctime_ns):
            raise ValueError("Runtime source changed while the archive was written")


def freeze(source_root, output):
    root = Path(source_root)
    output = Path(output)
    if (not root.is_absolute() or root.is_symlink()
            or root.resolve(strict=True) != root or output.exists() or output.is_symlink()):
        raise ValueError("Use an exact source root and a fresh archive path")
    payload, records, signatures = {}, [], {}
    for name in sorted(REQUIRED):
        if not _safe(name):
            raise ValueError("Required runtime closure contains an unsafe path")
        path = root / name
        if (path.is_symlink() or not path.is_file()
                or path.resolve(strict=True) != root / name):
            raise ValueError("Required runtime member is missing or indirect: " + name)
        before = path.stat()
        contents = path.read_bytes()
        after = path.stat()
        signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
                     before.st_ctime_ns)
        if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
                         after.st_ctime_ns) or not contents:
            raise ValueError("Runtime member changed or is empty: " + name)
        if name.endswith(".py"):
            compile(contents, name, "exec")
        digest = hashlib.sha256(contents).hexdigest()
        payload[name] = contents
        records.append({"path": name, "bytes": len(contents), "sha256": digest})
        signatures[name] = signature
    manifest = {"schema": SCHEMA, "files": records}
    manifest_raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                               allow_nan=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    incoming = output.with_name("." + output.name + ".incoming-" + uuid.uuid4().hex)
    failed = output.with_name("." + output.name + ".failed-" + uuid.uuid4().hex)
    published = False
    try:
        with incoming.open("xb") as destination:
            # Do not let GzipFile copy ``output.name`` into its header.  The runtime
            # archive identity must depend only on the reviewed member bytes.
            with gzip.GzipFile(filename="", fileobj=destination, mode="wb",
                               compresslevel=1, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as bundle:
                    for name in sorted(payload):
                        bundle.addfile(_tar_info(name, len(payload[name])), io.BytesIO(payload[name]))
                    bundle.addfile(_tar_info("code-manifest.json", len(manifest_raw)),
                                   io.BytesIO(manifest_raw))
            destination.flush(); os.fsync(destination.fileno())
        archive_sha256 = sha(incoming)
        _assert_source_stable(root, signatures)
        descriptor = os.open(output.parent, os.O_RDONLY)
        try:
            # Hard-link publication is an atomic create-if-absent operation on
            # this same filesystem.  It cannot overwrite a concurrently
            # created canonical archive after the initial freshness check.
            os.link(incoming, output); published = True
            incoming.unlink(); os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if sha(output) != archive_sha256:
            raise RuntimeError("Runtime archive changed across atomic publication")
    except Exception:
        evidence = output if published and output.exists() else incoming
        if evidence.exists() and not failed.exists():
            evidence.replace(failed)
            if published and incoming.exists():
                incoming.unlink()
            descriptor = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        raise
    return {"schema": "qwen38-native-gap-v4-runtime-freeze/v1", "status": "passed",
            "archive": str(output.resolve(strict=True)), "archive_sha256": archive_sha256,
            "code_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "file_count": len(records), "required_closure_complete": True,
            "deterministic_tar_metadata": True, "atomic_publish": True,
            "atomic_no_clobber_publish": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(freeze(**vars(parser.parse_args())), sort_keys=True))
