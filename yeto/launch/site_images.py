"""The site image catalog: images a site adopted for a model, with their smoke evidence.

``yeto image adopt`` (:mod:`.image_adopt`) checks an image built for one model
(manifest, capabilities, fit to the nodes, pre-pull, a smoke run) and records it
here. A trainer island or a sampler pool that names no image then takes the newest
verified record that matches its model and its nodes, before the built-in default.

The catalog is one YAML file, ``~/.yeto/site-images.yaml`` unless an SSH node
inventory names another with its top-level ``site_images:`` key::

    version: 1
    images:
      - model: moonshotai/Kimi-K3.1-Pro   # any spelling engine_images.model_key knows
        role: trainer                     # trainer | sampler
        engine: miles                     # trainer: the --rl-backend; sampler: vllm | sglang | vllm-ascend
        ref: ghcr.io/OWNER/IMAGE@sha256:<64 hex>
        device_family: cuda               # optional; checked when given
        cpu_arch: x86_64
        card_archs: [sm103]               # the card architectures the smoke ran on
        versions: {python: 3.12.3, torch: 2.9.0, cuda: "13.0"}
        evidence: {at: "2026-11-26T10:00:00Z", node: b300-0, seconds: 812.4, result: pass}
        status: verified-smoke            # verified-smoke | retired

A record matches (model, role, node profile) when its model key, role and status
``verified-smoke`` agree and every node of the island or pool has the record's CPU
architecture, one of its card architectures and (when the record names one) its
device family. Several matches: the newest ``evidence.at`` wins, then the later
record in the file.

The selection order (:func:`select`): an image the configuration names (``--rl-image``,
an inventory image, a pool's ``container.image``) as it is, without reading the
catalog; else the catalog's match; else the built-in default. A broken catalog is
refused (:class:`SiteImagesError`) only when it is read, never for a configuration
that names its image.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import engine_images

DEFAULT_PATH = "~/.yeto/site-images.yaml"
INVENTORY_KEY = "site_images"
VERSION = 1
VERIFIED = "verified-smoke"
RETIRED = "retired"
STATUSES = (VERIFIED, RETIRED)
ROLES = engine_images.ROLES
TRAINER_ENGINES = tuple(sorted({r.engine for r in engine_images.DEFAULT_IMAGES if r.role == "trainer"}))
SAMPLER_ENGINES = engine_images.ENGINES
SOURCE_EXPLICIT = "explicit"
SOURCE_SITE = "site image catalog"
SOURCE_DEFAULT = "built-in default"
_REQUIRED = ("model", "role", "engine", "ref", "cpu_arch", "card_archs", "status")
_OPTIONAL = ("device_family", "versions", "evidence")
_EVIDENCE_REQUIRED = ("at", "node", "seconds", "result")
_DIGEST_REF = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}")


class SiteImagesError(ValueError):
    """The site image catalog cannot be read or a record is not valid."""


@dataclass(frozen=True)
class SiteImage:
    """One record of the catalog (:data:`_REQUIRED` and :data:`_OPTIONAL` keys)."""

    model: str
    role: str
    engine: str
    ref: str
    cpu_arch: str
    card_archs: tuple[str, ...]
    status: str
    device_family: str | None = None
    versions: Mapping[str, str] = field(default_factory=dict)
    evidence: Mapping[str, Any] = field(default_factory=dict)

    @property
    def model_key(self) -> str:
        return engine_images.model_key(self.model)

    @property
    def at(self) -> str:
        return str(self.evidence.get("at") or "")

    def same_image(self, other: SiteImage) -> bool:
        """The same (model, role, engine, ref): a newer adoption replaces the older record."""
        return (self.model_key, self.role, self.engine, self.ref) == \
            (other.model_key, other.role, other.engine, other.ref)

    def problems(self, profile: Any) -> list[str]:
        """Why this record does not fit the node ``profile`` (empty: it does)."""
        facts = profile_facts(profile)
        out = []
        if facts["cpu_arch"] != self.cpu_arch:
            out.append(f"it is a {self.cpu_arch} image, node {facts['name']} is {facts['cpu_arch']}")
        if facts["card_arch"] not in self.card_archs:
            out.append(f"it was smoked on {'/'.join(self.card_archs)} cards, node {facts['name']} has "
                       f"{facts['card_arch'] or 'cards of an unknown architecture'}")
        if self.device_family and facts["family"] != self.device_family:
            out.append(f"it is a {self.device_family} image, node {facts['name']} is {facts['family']}")
        return out

    def to_doc(self) -> dict[str, Any]:
        doc: dict[str, Any] = {"model": self.model, "role": self.role, "engine": self.engine, "ref": self.ref}
        if self.device_family:
            doc["device_family"] = self.device_family
        doc.update({"cpu_arch": self.cpu_arch, "card_archs": list(self.card_archs)})
        if self.versions:
            doc["versions"] = dict(self.versions)
        if self.evidence:
            doc["evidence"] = dict(self.evidence)
        doc["status"] = self.status
        return doc

    def describe(self) -> str:
        """``<ref> (adopted <at> on <node>)`` for a dry-run line."""
        how = " ".join(x for x in (f"adopted {self.at}" if self.at else "",
                                    f"on {self.evidence['node']}" if self.evidence.get("node") else "") if x)
        return f"{self.ref}" + (f" ({how})" if how else "")


@dataclass(frozen=True)
class Choice:
    """What :func:`select` chose: the reference (None: no image at all), where it came
    from (:data:`SOURCE_EXPLICIT` / :data:`SOURCE_SITE` / :data:`SOURCE_DEFAULT`), the
    catalog record (site only), the catalog path read (None: not read) and why the
    catalog gave nothing (default only)."""

    ref: str | None
    source: str
    record: SiteImage | None = None
    path: Path | None = None
    why: str = ""

    def line(self, what: str) -> str:
        """A dry-run line: ``<what> image <ref> (source: ...)``."""
        if self.source == SOURCE_SITE:
            assert self.record is not None
            return f"{what} image {self.record.describe()} (source: {SOURCE_SITE} {self.path})"
        ref = self.ref or "(none)"
        if self.source == SOURCE_DEFAULT:
            return f"{what} image {ref} (source: {SOURCE_DEFAULT}; {self.why})"
        return f"{what} image {ref} (source: {self.source})"

    def __iter__(self):  # (ref, source), the contract's shape
        return iter((self.ref, self.source))


# --- paths -------------------------------------------------------------------------


def default_path() -> Path:
    return Path(os.path.expanduser(DEFAULT_PATH))


def catalog_path(value: str | os.PathLike[str] | None = None, *, base: Path | None = None) -> Path:
    """The catalog file: ``value`` (``~`` expanded; a relative path is taken from ``base``,
    the inventory's directory), else :data:`DEFAULT_PATH`."""
    if value is None or not str(value).strip():
        return default_path()
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute() and base is not None:
        path = Path(base) / path
    return path


def inventory_value(document: Any) -> str | None:
    """The inventory document's ``site_images:`` (None: not given). Refuses a non-string."""
    if not isinstance(document, Mapping) or document.get(INVENTORY_KEY) is None:
        return None
    value = document[INVENTORY_KEY]
    if not isinstance(value, str) or not value.strip():
        raise SiteImagesError(f"inventory {INVENTORY_KEY}: must be a file path, not {value!r}")
    return value


def inventory_catalog_path(inventory: str | os.PathLike[str] | None) -> Path:
    """The catalog of the SSH node inventory file ``inventory`` (its ``site_images:``,
    relative to the file's directory), else the default. An unreadable inventory reads as
    the default here: the inventory's own loader reports it."""
    if not inventory:
        return default_path()
    import yaml  # noqa: PLC0415

    file = Path(os.path.expanduser(str(inventory)))
    try:
        document = yaml.safe_load(file.read_text())
    except (OSError, yaml.YAMLError):
        return default_path()
    return catalog_path(inventory_value(document), base=file.parent)


# --- schema ------------------------------------------------------------------------


def _text(raw: Mapping[str, Any], key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SiteImagesError(f"{where}: {key} must be a non-empty string, not {value!r}")
    return value.strip()


def parse_record(raw: Any, *, where: str = "record") -> SiteImage:
    """One catalog record, validated (:class:`SiteImagesError` names the key)."""
    if not isinstance(raw, Mapping):
        raise SiteImagesError(f"{where}: must be a mapping, not {type(raw).__name__}")
    unknown = sorted(set(raw) - set(_REQUIRED) - set(_OPTIONAL))
    if unknown:
        raise SiteImagesError(f"{where}: unknown key(s) {unknown} (known: {sorted(_REQUIRED + _OPTIONAL)})")
    missing = [k for k in _REQUIRED if k not in raw]
    if missing:
        raise SiteImagesError(f"{where}: missing key(s) {missing}")
    model = _text(raw, "model", where)
    role = _text(raw, "role", where)
    if role not in ROLES:
        raise SiteImagesError(f"{where}: role {role!r} is unknown (known: {list(ROLES)})")
    engine = _text(raw, "engine", where).lower()
    engines = TRAINER_ENGINES if role == "trainer" else SAMPLER_ENGINES
    if engine not in engines:
        raise SiteImagesError(f"{where}: {role} engine {engine!r} is unknown (known: {list(engines)})")
    ref = _text(raw, "ref", where).removeprefix("docker:")
    if not _DIGEST_REF.fullmatch(ref):
        raise SiteImagesError(f"{where}: ref {ref!r} must be pinned by digest (REGISTRY/IMAGE@sha256:<64 hex>)")
    try:
        cpu_arch = engine_images.canonical_cpu_arch(_text(raw, "cpu_arch", where))
    except engine_images.UnknownCpuArch as error:
        raise SiteImagesError(f"{where}: {error}") from None
    archs = raw.get("card_archs")
    if isinstance(archs, str) or not isinstance(archs, (list, tuple)) or not archs \
            or not all(isinstance(a, str) and a.strip() for a in archs):
        raise SiteImagesError(f"{where}: card_archs must be a non-empty list of card architectures "
                              f"(sm103, gfx950, ascend910b ...), not {archs!r}")
    family = raw.get("device_family")
    if family is not None:
        try:
            family = engine_images.canonical_device_family(str(family))
        except ValueError as error:
            raise SiteImagesError(f"{where}: {error}") from None
    status = _text(raw, "status", where)
    if status not in STATUSES:
        raise SiteImagesError(f"{where}: status {status!r} is unknown (known: {list(STATUSES)})")
    versions = raw.get("versions") or {}
    if not isinstance(versions, Mapping):
        raise SiteImagesError(f"{where}: versions must be a mapping, not {type(versions).__name__}")
    evidence = raw.get("evidence") or {}
    if not isinstance(evidence, Mapping):
        raise SiteImagesError(f"{where}: evidence must be a mapping, not {type(evidence).__name__}")
    if status == VERIFIED:
        lacking = [k for k in _EVIDENCE_REQUIRED if evidence.get(k) in (None, "")]
        if lacking:
            raise SiteImagesError(f"{where}: a {VERIFIED} record needs evidence {lacking}")
        if str(evidence.get("result")) != "pass":
            raise SiteImagesError(f"{where}: a {VERIFIED} record needs evidence.result pass, "
                                  f"not {evidence.get('result')!r}")
    evidence = {str(k): _plain(v) for k, v in evidence.items()}
    return SiteImage(model=model, role=role, engine=engine, ref=ref, cpu_arch=cpu_arch,
                     card_archs=tuple(a.strip().lower() for a in archs), status=status, device_family=family,
                     versions={str(k): str(v) for k, v in versions.items()},
                     evidence=evidence)


def _plain(value: Any) -> Any:
    """A YAML timestamp (``at: 2026-11-26T10:00:00Z`` unquoted) as its ISO text, so records
    order by ``evidence.at`` and write back as they were read."""
    import datetime  # noqa: PLC0415

    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None and value.utcoffset() == datetime.timedelta(0):
            return value.replace(tzinfo=None).isoformat() + "Z"
        return value.isoformat()
    if isinstance(value, datetime.date):
        return value.isoformat()
    return value


def parse_catalog(document: Any, *, where: str = "site image catalog") -> list[SiteImage]:
    if document is None:
        return []
    if not isinstance(document, Mapping):
        raise SiteImagesError(f"{where}: must be a mapping with version and images, not {type(document).__name__}")
    unknown = sorted(set(document) - {"version", "images"})
    if unknown:
        raise SiteImagesError(f"{where}: unknown top-level key(s) {unknown} (known: ['images', 'version'])")
    if document.get("version", VERSION) != VERSION:
        raise SiteImagesError(f"{where}: version {document.get('version')!r} is not {VERSION}")
    images = document.get("images") or []
    if not isinstance(images, list):
        raise SiteImagesError(f"{where}: images must be a list, not {type(images).__name__}")
    return [parse_record(raw, where=f"{where}: images[{i}]") for i, raw in enumerate(images)]


def load(path: str | os.PathLike[str] | None = None) -> list[SiteImage]:
    """Every record of the catalog at ``path`` (default :data:`DEFAULT_PATH`); a missing
    file is an empty catalog, a broken one :class:`SiteImagesError`."""
    import yaml  # noqa: PLC0415

    file = catalog_path(path)
    try:
        text = file.read_text()
    except FileNotFoundError:
        return []
    except OSError as error:
        raise SiteImagesError(f"cannot read the site image catalog {file}: {error}") from None
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise SiteImagesError(f"the site image catalog {file} is not valid YAML: {error}") from None
    return parse_catalog(document, where=str(file))


def save(records: Sequence[SiteImage], path: str | os.PathLike[str] | None = None) -> Path:
    """Write ``records`` to the catalog: a temporary file, flushed to disk, renamed over
    it, and the directory flushed, so a crash leaves the old catalog or the new one."""
    import yaml  # noqa: PLC0415

    file = catalog_path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump({"version": VERSION, "images": [r.to_doc() for r in records]}, sort_keys=False)
    fd, tmp = tempfile.mkstemp(prefix=f".{file.name}.", dir=str(file.parent))
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, file)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _fsync_dir(file.parent)
    return file


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # a directory some filesystems cannot flush
        pass
    finally:
        os.close(fd)


def lock_path(path: str | os.PathLike[str] | None = None) -> Path:
    """The lock file next to the catalog (``.<name>.lock``)."""
    file = catalog_path(path)
    return file.parent / f".{file.name}.lock"


@contextlib.contextmanager
def locked(path: str | os.PathLike[str] | None = None) -> Iterator[Path]:
    """Hold the catalog's exclusive lock (``flock`` on :func:`lock_path`): two
    ``yeto image adopt`` runs writing the same catalog take turns, so neither loses the
    other's record."""
    lock = lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield catalog_path(path)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def add(record: SiteImage | Mapping[str, Any], path: str | os.PathLike[str] | None = None) -> Path:
    """Add ``record`` (validated) to the catalog; it replaces a record of the same image
    (:meth:`SiteImage.same_image`). The read and the write happen under the catalog's
    lock (:func:`locked`). A broken catalog is refused, never overwritten."""
    new = record if isinstance(record, SiteImage) else parse_record(record)
    parse_record(new.to_doc())  # the record as it will be read back
    with locked(path):
        records = [r for r in load(path) if not r.same_image(new)]
        return save([*records, new], path)


# --- matching ----------------------------------------------------------------------


def profile_facts(profile: Any) -> dict[str, Any]:
    """name / unified family / CPU and card architecture of a node profile, a plain
    inventory ``Node`` (card architecture from its card) or any object with those fields."""
    from .node_inventory import card_arch_of  # noqa: PLC0415 - node_inventory imports engine_images

    family = getattr(profile, "family", None) or getattr(profile, "device_family", None)
    card_arch = getattr(profile, "card_arch", None)
    if not card_arch and getattr(profile, "card", None):
        try:
            card_arch = card_arch_of(profile.card)
        except Exception:  # noqa: BLE001 - an unknown card has no architecture
            card_arch = None
    return {"name": getattr(profile, "name", "?"),
            "family": engine_images.canonical_device_family(family) if family else None,
            "cpu_arch": engine_images.canonical_cpu_arch(profile.cpu_arch),
            "card_arch": str(card_arch).lower() if card_arch else None}


def _profiles(profile: Any) -> list[Any]:
    if isinstance(profile, (list, tuple)):
        return list(profile)
    return [profile]


def match(model: str | None, role: str, profile: Any, *, engine: str | None = None,
          records: Iterable[SiteImage] | None = None, path: str | os.PathLike[str] | None = None) -> list[SiteImage]:
    """The verified records for ``model`` in ``role`` that fit every node of ``profile``
    (one profile or a list: an island's or a pool's nodes), best first: the newest
    ``evidence.at``, then the later record in the file. ``engine``: only that engine's
    records. No model: nothing matches."""
    if role not in ROLES:
        raise ValueError(f"role {role!r} is unknown (known: {list(ROLES)})")
    if model is None or not str(model).strip():
        return []
    key = engine_images.model_key(model)
    nodes = _profiles(profile)
    found = []
    for index, record in enumerate(load(path) if records is None else records):
        if record.status != VERIFIED or record.role != role or record.model_key != key:
            continue
        if engine is not None and record.engine != str(engine).lower():
            continue
        if nodes and any(record.problems(node) for node in nodes):
            continue
        found.append((record.at, index, record))
    return [r for _, _, r in sorted(found, key=lambda t: (t[0], t[1]), reverse=True)]


def select(explicit: str | None, model: str | None, role: str, profile: Any, *, engine: str | None = None,
           path: str | os.PathLike[str] | None = None,
           default: str | Callable[[], str | None] | None = None) -> Choice:
    """The image of an island or a pool: ``explicit`` as it is (the catalog is not read),
    else the catalog's best match (:func:`match`), else ``default`` (a reference or a
    function that returns one; None: no default). Unpacks as ``(ref, source)``."""
    if explicit:
        return Choice(str(explicit), SOURCE_EXPLICIT)
    file = catalog_path(path)
    records = load(file)
    found = match(model, role, profile, engine=engine, records=records)
    if found:
        return Choice(found[0].ref, SOURCE_SITE, found[0], file)
    if not records:
        why = f"no catalog at {file}" if not file.exists() else f"{file} has no record"
    elif model is None or not str(model).strip():
        why = "no model to match"
    else:
        what = f"{engine_images.model_key(model)} {role}" + (f" ({engine})" if engine else "")
        why = f"no {VERIFIED} record in {file} for {what} on these nodes"
    ref = default() if callable(default) else default
    return Choice(ref, SOURCE_DEFAULT, None, file, why)
