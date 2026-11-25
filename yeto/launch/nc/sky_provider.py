"""SkyPilot lease provider: rollout nodes on clouds sky can launch (Nebius, AWS,
GCP, Kubernetes, ...), behind the same :class:`~yeto.launch.pool.provider.IslandProvider`
contract as National Compute. The pool's policy (eviction handling, fencing,
events, budget, breaker, scale) and the SSH + docker node route are unchanged;
only the supply differs. It also carries the machine management of the spot
watchdog (:mod:`yeto.rl.spot.watchdog`): the candidate rotation, the start
timeout, the own-name guard and the per-offer price.

* ``order(slot)``: submit ``sky.launch`` of a one-node cluster ``<prefix>-<run>-<n>``
  with the offer at the cursor (:mod:`yeto.launch.rollout.offers`: explicit
  ``offers``, the watchdog's Nebius ``candidates``, or one ``resources``) and
  return at once; the cluster name is the ticket and the launch's request id is
  kept in the record. The order is recorded locally (and in the run's resource
  ledger as ``sky_cluster``) *before* the launch. yeto's sky patches
  (:mod:`yeto.sky_patches`, the Nebius spot pricing among them) are installed
  before the first order;
* the launch request is confirmed with ``sky.get`` once sky reports it finished
  (``sky.api_status``; the pool tick never blocks on a launch): a success moves
  the cursor back to the cheapest offer, a failure (no capacity, quota, ...)
  cools that offer down and moves the cursor to the next one, and the ticket
  reads ended (``lost``, the error as ``detail``) so the pool releases it and
  orders again on the next offer (an offer failure is not the pool breaker's
  while another offer is open: ``has_fallback_offer``). The spot watchdog's
  blocking ``launch_cluster(wait=True)`` raises the failure instead;
* idle autodown: a cluster with no sky job (the sampler runs over SSH + docker)
  is launched with ``down=True``, ``wait_for=none`` and
  ``idle_minutes_to_autostop`` = ``hard_cap_minutes`` (default 24 h), so sky's
  idle check (no job = idle) never takes it down while the sampler runs; it
  goes at the hard cap after launch, or when the pool releases it. A task with
  a ``run`` command (``task_yaml``: the sampler is the sky job) keeps
  ``down=True`` alone: the cluster goes when that job ends;
* clouds sky cannot autodown (``NO_AUTODOWN_CLOUDS``: Verda and a few others;
  sky refuses a launch that asks for it there): the cluster is launched without
  ``down`` / autostop, and its active status carries ``lease_until`` = order +
  ``hard_cap_minutes``, so the pool's lease-end rules (a replacement before the
  end, the drain and release at it) stand in for sky's hard cap;
* ``status``: ``sky.status`` of that cluster (``refresh=AUTO``, so a reclaimed
  spot VM is seen): ``UP`` -> active, ``INIT`` (or the launch still running) ->
  waiting (ended ``lost`` past ``start_timeout_s`` from the order: the watchdog's
  "stuck in STARTING" rule), ``STOPPED`` / gone -> ended ``preempted`` (sky
  clusters end that way when the cloud reclaims spot capacity; there is no
  lease end, so ``lease_until`` is None and only eviction-driven replacement
  applies);
* API server restart: a local sky API server starts with an empty request
  table, so every request id the record holds is unknown afterwards. Such a
  launch is reconciled from what sky lists under its recorded cluster name: an
  ``UP`` cluster is kept (no new order), one still ``INIT`` (sky never resumes an
  interrupted request) or none at all reads ended ``lost`` and is replaced;
* ``release``: cancel a launch still running, then ``sky.down`` (only clusters this
  provider launched for this run: the name prefix *and* the local record),
  confirmed gone (``ended`` only once sky no longer lists it);
* ``node``: the head IP and ssh user from the cluster handle;
* ``run_tickets``: open records, plus clusters sky lists that are positively this
  run's (a launch the process did not live to record is adopted, then released).
  Ownership is a mark, not a name: a cluster is adopted only when its name is
  exactly ``<prefix>-<n>`` *and* it carries the sky label ``yeto-run=<owner>``
  (set on every launch) or this run's ledger holds an ``intent`` row for it with
  ``run=<run id>`` (written before the launch). Another run's cluster with a
  matching name (the spot watchdog's ``exact_prefix`` has no run id in it) or one
  opened by hand is never adopted, released or downed: it is reported once;
* ``attach``: an ssh alias to the head (sky's key, no relay), then the
  ``host_setup`` commands of the config;
* ``hourly_cost(ticket)``: the USD price of the offer the cluster was launched with.

"Accounts" are slots ``1..max_nodes``: one cluster per slot at a time.

Requires SkyPilot >= 0.14.3 (``sky.api_status`` / ``sky.api_cancel``). Pool config::

    "provider": {"kind": "sky",
                 "candidates": {"gpus": ["H100", "H200"], "spot_only": true, "infra": "nebius"},
                 "cluster_prefix": "yeto-smp", "ssh_key": "~/.sky/clients/<id>/ssh/sky-key",
                 "start_timeout_s": 1800, "hard_cap_minutes": 1440, "host_setup": ["sudo docker info"]}

(``"factory": "yeto.launch.nc.sky_provider:from_config"`` is the same provider.)

AWS credentials: the provider's sky requests never carry the run env's ``AWS_PROFILE`` /
``AWS_*`` keys (sky forwards them to a local API server, where they would override the
server's own; a run sets them for its object store). ``"aws_profile"`` names the AWS
profile sky's requests use. A pool whose offers may land on AWS (AWS or unpinned infra)
is refused, at ``pool --dry-run`` and again before its first order, while the run env
holds AWS settings a sky API server would apply: a profile without ``aws_profile``, key
variables (botocore prefers them to any profile), or endpoint / config file variables (a
server sky starts inherits them). Give the store ``YETO_RL_STORE_S3_PROFILE`` instead.
``aws_profile`` only reaches a local API server: a remote one (or one enforcing mount
containment) drops forwarded credentials and uses its own; there, sky's own
``workspaces.<ws>.aws.profile`` server config is the knob. The spot watchdog's provider
route has no ``aws_profile`` (its Nebius offers make no AWS call).
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from yeto.launch.rollout.offers import Offer, offers_from_config

from ..pool.provider import provider_caps
from ..pool.status import LeaseStatus
from .client import NodeLease
from .tunnel import state_dir

UP, INIT, STOPPED = "UP", "INIT", "STOPPED"
KIND, LEDGER_KIND = "sky", "sky_cluster"
CONFIG_KEYS = frozenset({"kind", "factory", "resources", "price_usd_h", "offers", "candidates", "cluster_prefix",
                         "ssh_key", "host_setup", "start_timeout_s", "task_yaml", "envs", "secrets",
                         "hard_cap_minutes", "aws_profile"})
# A cluster with no sky job is autodowned this long after its launch (wait_for=none).
DEFAULT_HARD_CAP_MINUTES = 24 * 60
# Clouds whose sky provisioner has no autodown (SkyPilot 0.14.3: CloudImplementationFeatures.AUTODOWN
# unsupported); sky refuses a launch with down=True or an autostop there.
NO_AUTODOWN_CLOUDS = frozenset({"hyperbolic", "primeintellect", "verda", "yotta"})
# sky request states (sky.server.requests.requests.RequestStatus) of a request still in flight
REQUEST_LIVE = frozenset({"PENDING", "WAITING", "RUNNING"})
# launch_state of a record row: the launch request is in flight / confirmed / failed /
# unknown to the API server (restarted) / cancelled by release
SUBMITTED, LAUNCHED, FAILED, LOST, CANCELLED = "submitted", "launched", "failed", "lost", "cancelled"


def _field(obj: Any, key: str) -> Any:
    """``obj[key]`` of a sky response: a dict, or a pydantic model (``StatusResponse``,
    ``RequestPayload``) read by attribute, not through sky's dict-compat layer."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
ERROR_TEXT_LIMIT = 600


def _error_text(error: BaseException, limit: int = ERROR_TEXT_LIMIT) -> str:
    """One line of ``error`` (no terminal colours). A longer one keeps its head and its tail:
    sky puts the per-zone reasons of a failed launch (a cloud's refusal, an unsupported
    feature) at the end of its message."""
    text = f"{type(error).__name__}: {' '.join(_ANSI.sub('', str(error)).split())}"
    if len(text) <= limit:
        return text
    head = limit // 3
    return f"{text[:head]} ... {text[-(limit - head - 5):]}"


def offer_cloud(offer: Offer) -> str:
    """The cloud an offer names (``cloud``, or the first part of ``infra``), lower case; '' when
    sky picks it."""
    res = offer.resources
    return str(res.get("cloud") or str(res.get("infra") or "").partition("/")[0]).strip().lower()


def _listed_cloud(record: Any) -> str:
    """The cloud of a cluster sky lists (its ``cloud`` field, or the handle's launched resources)."""
    cloud = record.get("cloud") or getattr(getattr(record.get("handle"), "launched_resources", None), "cloud", None)
    return str(cloud or "").strip().lower()


def autodown_supported(offer: Offer) -> bool:
    """False when sky cannot autodown a cluster of this offer's cloud. An offer without a
    cloud keeps autodown: sky's optimizer then only picks clouds that have it."""
    return offer_cloud(offer) not in NO_AUTODOWN_CLOUDS


def _interrupted(error: BaseException) -> bool:
    """The API server lost the request (restarted): not the launch's own failure."""
    name = type(error).__name__
    text = str(error)
    return name == "RequestInterruptedError" or (
        isinstance(error, RuntimeError) and text.startswith("Failed to get request"))


def _status_name(value: Any) -> str:
    return str(getattr(value, "name", value)).rsplit(".", 1)[-1].upper()


OWNER_LABEL = "yeto-run"

# The AWS credential variables sky's client forwards to a local API server with every
# request (sky.server.requests.payloads.EXTERNAL_LOCAL_ENV_VARS), where they override
# the server's own. A run's env sets AWS_PROFILE for its object store (an R2 profile,
# say): forwarded, sky would call AWS with it. SkyProvider's requests carry none of
# them; ``aws_profile`` (provider config) is the profile its requests use instead.
AWS_FORWARDED_ENVS = ("AWS_PROFILE", "AWS_DEFAULT_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY",
                      "AWS_SESSION_TOKEN")
AWS_PROFILE_ENVS = ("AWS_PROFILE", "AWS_DEFAULT_PROFILE")  # replaced by ``aws_profile``
AWS_KEY_ENVS = ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN")  # beat any profile in botocore
# Not forwarded per request, but inherited by a local API server sky starts from this
# process (os.environ), where botocore applies them to sky's AWS calls
AWS_INHERITED_ENVS = ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_STS", "AWS_ENDPOINT_URL_EC2",
                      "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE")
_REQUEST_ENV: contextvars.ContextVar[dict[str, str | None] | None] = contextvars.ContextVar(
    "yeto_sky_request_env", default=None)  # copied into sky's own worker threads (run_in_parallel)
_SENTINEL = "yeto-sky-request-env-self-check"


def aws_request_env(aws_profile: str | None) -> dict[str, str | None]:
    """The request env overrides of a provider: every forwarded AWS variable dropped,
    ``AWS_PROFILE`` set to ``aws_profile`` when given."""
    env: dict[str, str | None] = {name: None for name in AWS_FORWARDED_ENVS}
    if aws_profile:
        env["AWS_PROFILE"] = aws_profile
    return env


def _scoped_request_env(original):
    """sky's ``request_body_env_vars`` with the calling SkyProvider's overrides applied
    (None drops the variable); other callers' requests are left as sky builds them."""
    @functools.wraps(original)
    def request_body_env_vars(*args: Any, **kw: Any) -> dict:
        env = dict(original(*args, **kw))
        for name, value in (_REQUEST_ENV.get() or {}).items():
            if value is None:
                env.pop(name, None)
            else:
                env[name] = value
        return env

    request_body_env_vars.yeto_scoped = True  # type: ignore[attr-defined]
    return request_body_env_vars


class ScopeUnavailable(RuntimeError):
    """This sky cannot have its request env scoped (a deterministic failure: no retry)."""


def install_request_env_scope() -> None:
    """Wrap sky's request env builder once (idempotent), then prove it is in effect: a
    request body built in a scope carries the scope's sentinel. Raises RuntimeError when
    sky has no such builder or does not use it (a sky version that moved it:
    :class:`ScopeUnavailable`, final) or the self-check itself raised (a plain
    RuntimeError, tried again later): the provider then refuses to order rather than
    forward the run's AWS credentials."""
    try:
        from sky.server.requests import payloads  # noqa: PLC0415
    except ImportError as error:
        raise ScopeUnavailable(f"sky has no request payloads module ({error})") from error
    current = getattr(payloads, "request_body_env_vars", None)
    if current is None:
        raise ScopeUnavailable("sky has no payloads.request_body_env_vars to scope the request env")
    if not getattr(current, "yeto_scoped", False):
        payloads.request_body_env_vars = _scoped_request_env(current)
    token = _REQUEST_ENV.set({"AWS_PROFILE": _SENTINEL})
    try:
        seen = payloads.RequestBody().env_vars.get("AWS_PROFILE")
    except Exception as error:  # noqa: BLE001
        raise RuntimeError(f"sky request env self-check failed ({error!r:.160})") from error
    finally:
        _REQUEST_ENV.reset(token)
    if seen != _SENTINEL:
        raise ScopeUnavailable("sky builds its request env without payloads.request_body_env_vars: the run's "
                               "AWS credentials cannot be kept out of sky's requests on this sky version")


@contextlib.contextmanager
def request_env_scope(overrides: dict[str, str | None]):
    """Sky requests built inside (this thread, and work sky hands to threads with a
    copied context) carry ``overrides``."""
    token = _REQUEST_ENV.set(overrides)
    try:
        yield
    finally:
        _REQUEST_ENV.reset(token)


class _ScopedSky:
    """The sky module, every SDK call made inside the provider's request env scope.
    ``refuse``: every SDK call raises it instead (no scope, and an unscoped request
    would carry the wrong AWS credentials)."""

    def __init__(self, module: Any, overrides: dict[str, str | None], refuse: str | None = None):
        self._module, self._overrides, self._refuse = module, overrides, refuse

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._module, name)
        if not callable(value) or isinstance(value, type):
            return value

        @functools.wraps(value)
        def call(*args: Any, **kw: Any) -> Any:
            if self._refuse:
                raise RuntimeError(self._refuse)
            with request_env_scope(self._overrides):
                return value(*args, **kw)

        return call


def unscoped_refusal(aws_profile: str | None, environ=None, what: str = "sky requests",
                     cluster: str | None = None) -> str | None:
    """Without the request env scope, a sky request would carry this process's AWS
    variables. Refused when they or ``aws_profile`` say another account than the
    server's own: the message gives the manual command."""
    environ = os.environ if environ is None else environ
    forwarded = [n for n in AWS_FORWARDED_ENVS if environ.get(n)]
    if not aws_profile and not forwarded:
        return None
    target = cluster or "<cluster>"
    manual = (f"AWS_PROFILE={aws_profile} sky down {target}" if aws_profile else
              f"env -u {' -u '.join(forwarded)} sky down {target}")
    return (f"sky provider: {what} refused without the request env scope (they would carry "
            f"{', '.join(forwarded) or 'no AWS_PROFILE'} instead of "
            f"{'aws_profile ' + aws_profile if aws_profile else 'the server credentials'}); by hand: {manual}")


def owner_token(run_id: str) -> str:
    """The ``yeto-run`` label value of ``run_id``: the id itself when every cloud's label
    rules take it (GCP's are the strictest), else a hash of it."""
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,62}", run_id):
        return run_id
    return "h-" + hashlib.sha256(run_id.encode()).hexdigest()[:24]


def run_prefix(cluster_prefix: str, run_id: str) -> str:
    """``<cluster_prefix>-<run>``: a run id that is not already lower-case letters, digits
    and '-' gets a hash of the original appended, so two ids that normalize alike
    (``ABC_2`` and ``abc-2``) never share a prefix."""
    slug = re.sub(r"[^a-z0-9-]", "-", run_id.lower())
    if slug != run_id:
        slug += "-" + hashlib.sha256(run_id.encode()).hexdigest()[:8]
    return f"{cluster_prefix}-{slug}"


def _record_ledger(fn: str, *args: Any, **kw: Any) -> None:
    from yeto.cost.ledger import record_safely  # noqa: PLC0415

    record_safely(fn, *args, **kw)


class SkyProvider:
    kind = KIND
    ledger_kind = LEDGER_KIND
    caps = provider_caps(KIND)  # yeto/cloud/capabilities.json

    def __init__(self, run_id: str, *, slots: list[int], resources: dict[str, Any] | None = None,
                 offers: list[Offer] | None = None, cluster_prefix: str = "yeto-smp", ssh_key: str | None = None,
                 host_setup: list[str] | None = None, start_timeout_s: float = 1800.0,
                 task_yaml: str | None = None, envs: dict[str, str] | None = None,
                 secrets: dict[str, str] | None = None, sky: Any = None, record: Path | None = None,
                 clock=time.time, exact_prefix: str | None = None,
                 hard_cap_minutes: float = DEFAULT_HARD_CAP_MINUTES, aws_profile: str | None = None):
        if not re.fullmatch(r"[a-z][a-z0-9-]{0,30}", cluster_prefix):
            raise ValueError("cluster_prefix: lower-case letters, digits and '-'")
        if offers is None:
            offers = offers_from_config({"resources": resources})
        if not offers:
            raise ValueError("sky provider: no offer")
        self.run_id, self.slots, self.offers = run_id, list(slots), list(offers)
        self.resources = dict(self.offers[0].resources)
        # exact_prefix: the spot watchdog's own cluster prefix (its names, its own-name guard)
        self.prefix = exact_prefix or run_prefix(cluster_prefix, run_id)
        self.owner = owner_token(run_id)
        self._name_re = re.compile(re.escape(self.prefix) + r"-(\d+)")
        self.ssh_key = os.path.expanduser(ssh_key) if ssh_key else None
        self.host_setup = list(host_setup or ())
        self.start_timeout_s = float(start_timeout_s)
        if not float(hard_cap_minutes) >= 1:
            raise ValueError("sky provider: hard_cap_minutes must be >= 1")
        self.hard_cap_minutes = int(hard_cap_minutes)
        self._patched = False
        self.aws_profile = aws_profile or None
        self.request_env = aws_request_env(self.aws_profile)
        self.task_yaml = os.path.expanduser(task_yaml) if task_yaml else None
        self.envs, self.secrets = dict(envs or {}), dict(secrets or {})
        self._sky, self.clock = sky, clock
        self._real_sky = sky is None  # imported on first use (not a test's fake)
        self.record = record or state_dir() / f"sky-clusters-{self.prefix}.json"

    # -- sky
    @property
    def sky(self):
        """The sky SDK; every call sends :attr:`request_env` (no run AWS credentials)."""
        if self._sky is None:
            import sky  # noqa: PLC0415

            self._sky = sky
        refuse = None
        if self._real_sky and not getattr(self, "_env_scoped", False):
            try:
                install_request_env_scope()
                self._env_scoped, self._scope_error = True, None
            except RuntimeError as error:
                # launches are refused (_install_patches). A version that cannot be scoped is final;
                # a self-check that raised is tried again on the next call
                self._env_scoped = isinstance(error, ScopeUnavailable)
                if getattr(self, "_scope_error", None) != str(error):
                    print(f"[sky-provider] WARNING: {error}; new launches are refused", flush=True)
                self._scope_error = str(error)
        if self._real_sky and getattr(self, "_scope_error", None):
            # status / down / cancel run unscoped only when that sends no other account's AWS
            # credentials; else they raise (a status error is never read as the cluster ended)
            refuse = unscoped_refusal(self.aws_profile, what="sky status / down / cancel")
        return _ScopedSky(self._sky, self.request_env, refuse)

    def _doc(self) -> dict[str, Any]:
        try:
            doc = json.loads(self.record.read_text())
        except (OSError, ValueError):
            return {"clusters": {}, "cursor": 0, "run": self.run_id}
        if "clusters" not in doc:  # the skeleton's flat {name: row} record
            doc = {"clusters": doc, "cursor": 0}
        if doc.setdefault("run", self.run_id) != self.run_id:
            raise PermissionError(f"sky provider: record {self.record} belongs to run {doc['run']!r}, "
                                  f"not {self.run_id!r}")
        return doc

    def _rows(self) -> dict[str, dict[str, Any]]:
        return self._doc()["clusters"]

    def _save_doc(self, doc: dict[str, Any]) -> None:
        self.record.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.record.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(self.record)

    def _named_ours(self, name: str) -> bool:
        return self._name_re.fullmatch(name) is not None

    def _ours(self, ticket: str) -> bool:
        return self._named_ours(ticket) and ticket in self._rows()

    def _marked_ours(self, record: Any) -> bool:
        """A positive ownership mark on a cluster sky lists: our ``yeto-run`` label on its
        launched resources, or an ``intent`` of this run in the run's resource ledger."""
        labels = _field(record, "labels") or getattr(getattr(_field(record, "handle"), "launched_resources", None),
                                                     "labels", None) or {}
        if labels.get(OWNER_LABEL) == self.owner:
            return True
        from yeto.cost.ledger import current  # noqa: PLC0415

        ledger = current()
        if ledger is None:
            return False
        try:
            rows = ledger.records()
        except Exception:  # noqa: BLE001 - an unreadable ledger is no mark
            return False
        name = _field(record, "name")
        return any(r.get("op") == "intent" and r.get("kind") == LEDGER_KIND and r.get("name") == name
                   and (r.get("attrs") or {}).get("run") == self.run_id for r in rows)

    def _record(self, ticket: str, *, refresh: bool = False) -> Any:
        """sky's record of cluster ``ticket`` (None: sky does not list it). ``refresh``:
        sky asks the cloud for clusters with spot instances or autostop (a reclaimed
        spot VM reads gone, not a stale UP)."""
        kw = {}
        modes = getattr(self.sky, "StatusRefreshMode", None)
        if refresh and modes is not None:
            kw["refresh"] = modes.AUTO
        records = self.sky.get(self.sky.status([ticket], **kw))
        return records[0] if records else None

    def cluster_state(self, ticket: str) -> str | None:
        """sky's status name of cluster ``ticket`` (``UP``, ``INIT``, ``STOPPED``), refreshed
        from the cloud; None when sky does not list it. Raises on a failed call."""
        record = self._record(ticket, refresh=True)
        return None if record is None else _status_name(_field(record, "status"))

    def _wait_for_none(self):
        modes = getattr(self.sky, "AutostopWaitFor", None)
        if modes is None:
            from sky.skylet.autostop_lib import AutostopWaitFor as modes  # noqa: PLC0415,N813
        return modes.NONE

    def launch_kwargs(self, task, *, autodown: bool = True) -> dict[str, Any]:
        """``sky.launch`` keywords. The sampler is no sky job (SSH + docker): sky's idle
        check would take the cluster down about a minute after the launch, so the idle
        timer waits for nothing and runs to the hard cap. A task with its own ``run``
        is the sampler as a sky job: down when it ends. ``autodown`` False (a cloud sky
        cannot autodown): none of them; :meth:`status` gives the hard cap as ``lease_until``."""
        if not autodown:
            return {}
        if getattr(task, "run", None):
            return {"down": True}
        return {"down": True, "idle_minutes_to_autostop": self.hard_cap_minutes,
                "wait_for": self._wait_for_none()}

    def _install_patches(self) -> None:
        """yeto's sky patches before the first order (the Nebius spot pricing among
        them), not only when ``yeto launch`` ran first. SkyPilot 0.14.3 sets the
        Nebius pricing model itself; on 0.13.0 the API server needs the hook too."""
        if self._patched:
            return
        from yeto import sky_patches  # noqa: PLC0415
        from yeto.sky_patches import nebius as nebius_patch  # noqa: PLC0415

        if self._real_sky:  # its API server must run this Python's sky version
            sky_patches.require_matching_api_server()
            problems = aws_env_problems(self, dict(os.environ))  # any orderer, not only a checked pool start
            self.sky  # noqa: B018 - installs the request env scope (or records why it could not)
            if getattr(self, "_scope_error", None):
                problems.append(f"sky provider: refusing to launch: {self._scope_error}")
            if problems:
                raise ValueError(problems[0])
        sky_patches.install()
        self._patched = True
        version = getattr(self.sky, "__version__", None)
        wants_nebius = any("nebius" in str(o.resources.get("infra") or o.resources.get("cloud") or "").lower()
                           and o.resources.get("use_spot") for o in self.offers)
        if wants_nebius and version in nebius_patch.SERVER_HOOK_SKY_VERSIONS:
            from yeto.launch.verda import remote_sky_api_server  # noqa: PLC0415
            from yeto.launcher import REPO_ROOT  # noqa: PLC0415

            if remote_sky_api_server():
                print(f"[sky-provider] WARNING: sky {version} on a remote API server: the Nebius spot pricing "
                      "patch is not in effect there", flush=True)
                return
            path, fresh = sky_patches.ensure_local_pth(str(REPO_ROOT))
            if fresh:
                print(f"[sky-provider] WARNING: sky patch hook {path} was only now installed; restart the sky API "
                      "server (sky api stop) before Nebius spot orders", flush=True)

    def _task(self, name: str, offer: Offer, num_nodes: int = 1):
        sky = self.sky
        task = sky.Task.from_yaml(self.task_yaml) if self.task_yaml else sky.Task(name=name, run=None)
        if num_nodes > 1:  # a group order (order_group): one cluster of num_nodes nodes
            task.num_nodes = int(num_nodes)
        if self.envs:
            task.update_envs(dict(self.envs))
        if self.secrets:
            task.update_secrets({k: os.environ.get(v, "") for k, v in self.secrets.items()})
        resources = dict(offer.resources)
        if resources.get("infra") or resources.get("cloud"):
            # sky keeps the labels in the cluster handle on every cloud (AWS / GCP / Kubernetes
            # also tag the instances). Without a cloud, sky validates labels against every
            # enabled cloud (a client-side cloud check): the ledger intent is the mark then.
            resources["labels"] = {**dict(resources.get("labels") or {}), OWNER_LABEL: self.owner}
        task.set_resources(sky.Resources(**resources))
        return task

    @property
    def cursor(self) -> int:
        return int(self._doc().get("cursor", 0)) % len(self.offers)

    # -- IslandProvider
    def order(self, account: int) -> str:
        self.last_offer_key = None  # set again once this order picked its offer
        doc = self._doc()
        n = 1 + max([int(r.get("n", 0)) for r in doc["clusters"].values()] or [0])
        offer = self._pick(doc)
        return self.launch_cluster(f"{self.prefix}-{n}", offer, slot=account, n=n, wait=False)

    # -- the per-offer breaker: an order is a (slot, offer); a failing offer cools down
    # alone while the others keep ordering (the pool's breaker counts only when none is left)
    OFFER_COOLDOWN_S = 600.0

    def _cooling(self, doc: dict[str, Any], key: str, now: float) -> bool:
        return float((doc.get("offers") or {}).get(key, {}).get("open_until", 0)) > now

    def _pick(self, doc: dict[str, Any]) -> Offer:
        """From the cursor on, the first offer not cooling down; all cooling: the one that reopens first."""
        now, start = self.clock(), int(doc.get("cursor", 0)) % len(self.offers)
        ring = self.offers[start:] + self.offers[:start]
        for offer in ring:
            if not self._cooling(doc, offer.key, now):
                return offer
        return min(ring, key=lambda o: float(doc["offers"][o.key]["open_until"]))

    def _offer_failed(self, doc: dict[str, Any], key: str) -> None:
        row = doc.setdefault("offers", {}).setdefault(key, {"fails": 0})
        row["fails"] = int(row.get("fails", 0)) + 1
        row["open_until"] = self.clock() + min(self.OFFER_COOLDOWN_S * 2 ** (row["fails"] - 1),
                                               8 * self.OFFER_COOLDOWN_S)

    def has_fallback_offer(self) -> bool:
        """True while some offer is not cooling down: an order failure is that offer's, not the pool's."""
        doc, now = self._doc(), self.clock()
        return len(self.offers) > 1 and any(not self._cooling(doc, o.key, now) for o in self.offers)

    def offer_of(self, ticket: str) -> str | None:
        """The offer key cluster ``ticket`` was ordered with (None: not ours)."""
        return ((self._rows().get(ticket) or {}).get("offer") or {}).get("key")

    def offer_breakers(self) -> dict[str, dict[str, Any]]:
        return dict(self._doc().get("offers") or {})

    def launch_cluster(self, name: str, offer: Offer, *, slot: int, n: int = 0, wait: bool = True,
                       num_nodes: int = 1) -> str:
        """Record, then launch cluster ``name`` (``<prefix>-...``) with ``offer``.
        ``wait``: block on the launch request (the spot watchdog); a failure moves
        the cursor past the offer and is raised. ``wait=False`` (the pool's ``order``):
        return once submitted; :meth:`status` confirms the request."""
        if not self._named_ours(name):
            raise PermissionError(f"cluster {name} is not named {self.prefix}-<n>")
        self._install_patches()
        autodown = autodown_supported(offer)
        doc = self._doc()
        doc["clusters"][name] = {"slot": slot, "n": n, "open": True, "ordered_at": self.clock(),
                                 "offer": offer.record(), "launch_state": SUBMITTED,
                                 **({"num_nodes": num_nodes} if num_nodes > 1 else {}),
                                 **({} if autodown else {"autodown": False})}
        self._save_doc(doc)  # recorded before the launch: a crash mid-launch still leaves it ours to down
        # pool / aws_profile: a crash before "created" leaves a row the cleanup still downs as this pool's
        _record_ledger("intent", LEDGER_KIND, name, offer=offer.key, price_usd_h=offer.price_usd_h, run=self.run_id,
                       pool=True, **({"aws_profile": self.aws_profile} if self.aws_profile else {}))
        self.last_offer_key = offer.key  # the pool's order_failed event names it (offer ledger)
        try:
            task = self._task(name, offer, num_nodes)
            request = self.sky.launch(task, cluster_name=name, **self.launch_kwargs(task, autodown=autodown))
        except Exception as error:  # the submit itself failed (bad resources, API server down)
            self._launch_failed(name, error)
            raise
        doc = self._doc()
        doc["clusters"][name]["launch_request"] = str(request)
        self._save_doc(doc)
        if not wait:
            return name
        try:
            self.sky.get(request)
        except Exception as error:
            self._launch_failed(name, error)
            raise
        self._launch_ok(name)
        return name

    # -- group order (yeto.launch.pool.group): one multi-node cluster; sky's best network
    # tier when the group asks for an RDMA fabric
    fabrics = ("tcp", "rdma")

    def order_group(self, n: int, shape: Any, fabric: str = "tcp") -> list[tuple[int, str]]:
        """Place ``n`` nodes as one cluster ``<prefix>-<k>`` on a free slot: one ticket, from
        the first offer (cursor order, cooling offers last) whose accelerators are the
        shape's. An RDMA fabric takes the cloud's best network tier for that shape
        (``topology.multinode_network_tier``; a shape the cloud has no such tier for is
        refused). A launch that fails releases what it left (``group.release_placed``)."""
        from ..pool.group import release_placed
        from ..topology import multinode_network_tier

        self.last_offer_key = None  # set again once this order picked its offer
        free = [s for s in self.slots if self.account_free(s)]
        if not free:
            raise RuntimeError(f"sky provider: no free slot for a group of {n}")
        doc = self._doc()
        offer = self._group_offer(doc, shape)
        if fabric != "tcp" and int(n) > 1:
            cloud = str(offer.resources.get("cloud") or str(offer.resources.get("infra") or "").partition("/")[0])
            tier = multinode_network_tier(cloud.lower(), str(shape.gpu), int(shape.gpus_per_node))
            if tier is None:
                raise ValueError(f"sky provider: {cloud} has no RDMA network tier for "
                                 f"{shape.gpus_per_node} x {shape.gpu} nodes; nothing ordered")
            offer = Offer(offer.key, {**offer.resources, "network_tier": tier}, offer.price_usd_h)
        k = 1 + max([int(r.get("n", 0)) for r in doc["clusters"].values()] or [0])
        name = f"{self.prefix}-{k}"
        try:
            self.launch_cluster(name, offer, slot=free[0], n=k, wait=False, num_nodes=int(n))
        except Exception as error:
            left = [(free[0], name)] if (self._rows().get(name) or {}).get("open") else []
            raise release_placed(self, left, error) from error
        return [(free[0], name)]

    def _group_offer(self, doc: dict[str, Any], shape: Any) -> Offer:
        """The offer a group of ``shape`` nodes orders: cursor order, an offer cooling down
        only when every matching one is."""
        want = (str(shape.gpu).upper(), int(shape.gpus_per_node))

        def accelerators(offer: Offer) -> tuple[str, int] | None:
            acc = offer.resources.get("accelerators")
            if not isinstance(acc, str) or not acc:
                return None
            gpu, _, count = acc.partition(":")
            return gpu.upper(), int(count or 1)

        start = int(doc.get("cursor", 0)) % len(self.offers)
        ring = [o for o in self.offers[start:] + self.offers[:start] if accelerators(o) == want]
        if not ring:
            raise ValueError(f"sky provider: no offer of {want[1]} x {want[0]} nodes "
                             f"(offers: {[o.resources.get('accelerators') for o in self.offers]}); nothing ordered")
        now = self.clock()
        open_ = [o for o in ring if not self._cooling(doc, o.key, now)]
        return open_[0] if open_ else min(ring, key=lambda o: float(doc["offers"][o.key]["open_until"]))

    def group_failed(self, ticket: str) -> None:
        """A group on cluster ``ticket`` did not come up: its offer cools down (once; a start
        timeout already counted it), so the next group order takes the next offer."""
        doc = self._doc()
        row = doc["clusters"].get(ticket) or {}
        key = (row.get("offer") or {}).get("key")
        if key and not row.get("start_timed_out") and not row.get("launch_failed") and not row.get("group_failed"):
            row["group_failed"] = True
            self._offer_failed(doc, key)
            doc["cursor"] = (next((i for i, o in enumerate(self.offers) if o.key == key), 0) + 1) % len(self.offers)
            self._save_doc(doc)

    def _launch_ok(self, name: str) -> None:
        doc = self._doc()
        row = doc["clusters"][name]
        if row.get("launch_state") == LAUNCHED:
            return
        row["launch_state"] = LAUNCHED
        key = (row.get("offer") or {}).get("key")
        doc["cursor"] = 0  # back to the cheapest offer for the next order
        (doc.get("offers") or {}).pop(key, None)
        self._save_doc(doc)
        offer = row.get("offer") or {}
        extra = {"aws_profile": self.aws_profile} if self.aws_profile else {}  # the cleanup's sky down uses it
        _record_ledger("created", LEDGER_KIND, name, name, offer=key, price_usd_h=offer.get("price_usd_h"),
                       run=self.run_id, slot=row.get("slot"), pool=True,
                       **({"num_nodes": row["num_nodes"]} if row.get("num_nodes") else {}), **extra)

    def _launch_failed(self, name: str, error: BaseException) -> None:
        """The launch failed: the next order tries the next offer; a cluster it left
        stays open for the pool to release, none frees the slot at once."""
        try:  # a launch that left no cluster bills nothing: the slot is free again
            left = self._record(name) is not None
        except Exception:  # noqa: BLE001 - unverified: keep it open for the pool to release
            left = True
        doc = self._doc()
        row = doc["clusters"][name]
        key = (row.get("offer") or {}).get("key")
        cursor = next((i for i, o in enumerate(self.offers) if o.key == key), 0)
        doc["cursor"] = (cursor + 1) % len(self.offers)
        if key:
            self._offer_failed(doc, key)
        row.update(launch_failed=True, launch_state=FAILED, launch_error=_error_text(error), open=left)
        self._save_doc(doc)

    def _request_live(self, request: str) -> bool | None:
        """True: in flight; False: finished (``sky.get`` returns at once); None: the
        API server does not know it (restarted: its request table starts empty)."""
        rows = self.sky.api_status([request])
        if not rows:
            return None
        return _status_name(_field(rows[0], "status")) in REQUEST_LIVE

    def _settle(self, ticket: str) -> dict[str, Any]:
        """Confirm the launch request of ``ticket`` once sky reports it finished; the row."""
        row = self._rows().get(ticket) or {}
        request = row.get("launch_request")
        if row.get("launch_state") != SUBMITTED or not request:
            return row
        live = self._request_live(request)  # raises on a failed call: a control-plane blip
        if live:
            return row
        if live is None:
            return self._mark(ticket, launch_state=LOST)
        try:
            self.sky.get(request)
        except Exception as error:  # noqa: BLE001 - the launch's own failure, or a lost request
            if _interrupted(error):
                return self._mark(ticket, launch_state=LOST)
            self._launch_failed(ticket, error)
            return self._rows().get(ticket) or {}
        self._launch_ok(ticket)
        return self._rows().get(ticket) or {}

    def _mark(self, ticket: str, **fields: Any) -> dict[str, Any]:
        doc = self._doc()
        doc["clusters"][ticket].update(fields)
        self._save_doc(doc)
        return doc["clusters"][ticket]

    def _start_timed_out(self, ticket: str) -> bool:
        """Past ``start_timeout_s`` from the order (counted once against its offer)."""
        ordered = (self._rows().get(ticket) or {}).get("ordered_at")
        if ordered is None or self.clock() - float(ordered) <= self.start_timeout_s:
            return False
        doc = self._doc()
        row = doc["clusters"].get(ticket) or {}
        if not row.get("start_timed_out"):
            row["start_timed_out"] = True
            key = (row.get("offer") or {}).get("key")
            if key:
                self._offer_failed(doc, key)
            self._save_doc(doc)
        return True

    def status(self, account: int, ticket: str) -> LeaseStatus:
        row = self._settle(ticket)  # raises on a failed call: a control-plane blip to the pool
        if row.get("launch_failed"):
            return LeaseStatus("ended", end_reason="lost", detail=row.get("launch_error"))
        state_of_launch = row.get("launch_state")
        record = self._record(ticket, refresh=True)
        if record is None:
            if state_of_launch == SUBMITTED:  # sky has not created the cluster yet
                if self._start_timed_out(ticket):
                    return LeaseStatus("ended", end_reason="lost", detail="launch still running past start_timeout_s")
                return LeaseStatus("waiting")
            if state_of_launch == LOST:
                return LeaseStatus("ended", end_reason="lost",
                                   detail="launch request lost by the sky API server before the cluster existed")
            return LeaseStatus("ended", end_reason="preempted")
        state = _status_name(_field(record, "status"))
        if state == UP:
            if state_of_launch == LOST:  # it came up before the API server restart: kept
                self._mark(ticket, launch_state=LAUNCHED, reconciled=True)
            return LeaseStatus("active", lease_until=self._hard_cap_end(row))
        if state == INIT:
            if state_of_launch == LOST:  # sky never resumes an interrupted launch
                return LeaseStatus("ended", end_reason="lost",
                                   detail="launch request lost by the sky API server while the cluster was INIT")
            if self._start_timed_out(ticket):
                return LeaseStatus("ended", end_reason="lost")  # stuck starting: released, next offer
            return LeaseStatus("waiting")
        return LeaseStatus("ended", end_reason="preempted" if state == STOPPED else state.lower())

    def _hard_cap_end(self, row: dict[str, Any]) -> float | None:
        """The hard cap of a cluster sky does not autodown (its order + ``hard_cap_minutes``),
        as a lease end; None when sky autodowns it."""
        if row.get("autodown", True) or row.get("ordered_at") is None:
            return None
        return float(row["ordered_at"]) + 60.0 * self.hard_cap_minutes

    def release(self, account: int, ticket: str) -> dict[str, Any]:
        if not self._ours(ticket):
            raise PermissionError(f"cluster {ticket} was not launched by this provider for run {self.run_id}")
        row = self._rows()[ticket]
        request = row.get("launch_request")
        if row.get("launch_state") == SUBMITTED and request:
            self._cancel_launch(ticket, request)
        if self._record(ticket) is not None:
            try:
                self.sky.get(self.sky.down(ticket))
            except Exception as error:  # noqa: BLE001 - gone meanwhile is fine; else checked below
                if type(error).__name__ != "ClusterDoesNotExist":
                    raise
        if self._record(ticket) is not None:
            raise RuntimeError(f"cluster {ticket}: down sent but it still exists")
        doc = self._doc()
        doc["clusters"][ticket]["open"] = False
        self._save_doc(doc)
        _record_ledger("released", LEDGER_KIND, ticket, ticket)
        return {"ticket": ticket, "state": "ended"}

    def _cancel_launch(self, ticket: str, request: str) -> None:
        """A launch still in flight is cancelled before the down (else it may create the
        cluster after the down); one that cannot be cancelled fails the release (retried)."""
        try:
            self.sky.get(self.sky.api_cancel([request]))
        except Exception as error:  # noqa: BLE001 - finished meanwhile is fine
            if self._request_live(request):
                raise RuntimeError(f"cluster {ticket}: cannot cancel its launch request ({_error_text(error)})") \
                    from error
        self._mark(ticket, launch_state=CANCELLED)

    def node(self, account: int, ticket: str) -> NodeLease:
        record = self._record(ticket)
        handle = None if record is None else _field(record, "handle")
        ip = getattr(handle, "head_ip", None)
        if not ip:
            raise RuntimeError(f"cluster {ticket} has no head IP yet")
        user = getattr(handle, "ssh_user", None) or "ubuntu"
        return NodeLease(account, ticket, ip, user, "", None, None, None, None)

    def balance(self) -> int | None:
        return None  # clouds bill outside; the run's spend cap (Budget) is the guard

    def account_free(self, account: int) -> bool:
        return not any(r.get("open") and r.get("slot") == account for r in self._rows().values())

    def run_tickets(self) -> list[tuple[int, str]]:
        """Open records, plus clusters sky lists that the record lacks and that are
        positively this run's (named ``<prefix>-<n>`` and marked, see the module notes):
        recorded now, on a free slot, so the pool can adopt or release them. A cluster
        with a matching name but no mark is reported once and left alone."""
        doc = self._doc()
        rows = doc["clusters"]
        try:
            # sky >= 0.14 answers pydantic StatusResponse records (not dicts): fields by _field
            listed = [r for r in (self.sky.get(self.sky.status(None)) or ()) if _field(r, "name")]
        except Exception:  # noqa: BLE001 - an unanswered listing: the record alone
            listed = []
        changed = False
        for rec in listed:
            name = str(_field(rec, "name"))
            match = self._name_re.fullmatch(name)
            if match is None or name in rows:
                continue
            if not self._marked_ours(rec):
                seen = doc.setdefault("unowned", [])
                if name not in seen:
                    seen.append(name)
                    changed = True
                    print(f"[sky-provider] cluster {name} has this run's name pattern but no ownership mark "
                          f"({OWNER_LABEL}={self.owner} label or ledger intent): not adopted, left alone",
                          flush=True)
                continue
            busy = {r.get("slot") for r in rows.values() if r.get("open")}
            slot = next((s for s in self.slots if s not in busy), self.slots[0])
            rows[name] = {"slot": slot, "n": int(match.group(1)), "open": True,
                          "ordered_at": self.clock(), "adopted": True}
            if _listed_cloud(rec) in NO_AUTODOWN_CLOUDS:  # its hard cap counts from its launch
                launched = rec.get("launched_at")
                rows[name].update(autodown=False, **({"ordered_at": float(launched)} if launched else {}))
            changed = True
        if changed:
            self._save_doc(doc)
        return [(int(r["slot"]), name) for name, r in rows.items() if r.get("open")]

    # -- pricing (the pool's spend in USD; see the rollout controller notes)
    def hourly_cost(self, ticket: str | None = None) -> float | None:
        row = (self._rows().get(ticket) or {}) if ticket else {}
        price = (row.get("offer") or {}).get("price_usd_h", self.offers[self.cursor].price_usd_h)
        return None if price is None else float(price) * int(row.get("num_nodes", 1))

    # -- node route hook
    def attach(self, lease: NodeLease, *, ssh_run) -> str:
        """An ssh alias to the head (direct, sky's key), then ``host_setup``."""
        from . import node_pool  # noqa: PLC0415

        alias = node_pool.host_alias(lease.ticket)
        ours = state_dir() / "ssh_config"
        lines = [f"Host {alias}", f"  HostName {lease.ip}", f"  User {lease.user}",
                 "  StrictHostKeyChecking accept-new", f"  UserKnownHostsFile {state_dir() / 'known_hosts'}",
                 "  ServerAliveInterval 15", "  ServerAliveCountMax 4"]
        if self.ssh_key:
            lines += [f"  IdentityFile {self.ssh_key}", "  IdentitiesOnly yes"]
        block = "\n".join(lines) + "\n"
        node_pool.edit_ssh_config(ours, lambda blocks: blocks.__setitem__(alias, block))
        node_pool.include_ours(ours)  # without it ``ssh <alias>`` does not resolve on a fresh pool host
        for command in self.host_setup:
            out = ssh_run(alias, command)
            if out.returncode != 0:
                raise RuntimeError(f"host setup {command[:40]!r} on {lease.ticket}: exit {out.returncode}")
        return alias


def registry_factory(cfg: dict[str, Any], run_id: str, section: dict[str, Any]) -> SkyProvider:  # noqa: ARG001
    """The registered ``sky`` factory (``yeto.launch.pool.provider``)."""
    return from_config(cfg, run_id)


def from_config(cfg: dict[str, Any], run_id: str, *, sky: Any = None) -> SkyProvider:
    section = dict(cfg.get("provider") or {})
    unknown = set(section) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"sky provider: unknown keys {sorted(unknown)}")
    offers = offers_from_config(section)
    for key in ("kind", "factory", "resources", "price_usd_h", "offers", "candidates"):
        section.pop(key, None)
    from . import cli  # noqa: PLC0415

    return SkyProvider(run_id, slots=cli.slot_list(cfg), offers=offers, sky=sky, **section)


def offline_problems(cfg: dict[str, Any]) -> list[str]:
    """``pool --dry-run`` checks of a sky pool (no sky call)."""
    import importlib.util  # noqa: PLC0415

    try:
        provider = from_config(cfg, str(cfg.get("run") or cfg.get("shard_base")), sky=object())
    except (ValueError, TypeError, KeyError) as error:
        return [str(error)]
    problems = []
    if importlib.util.find_spec("sky") is None:
        problems.append("the 'skypilot' Python package (module sky) is not installed")
    if provider.task_yaml and not Path(provider.task_yaml).exists():
        problems.append(f"sky provider: task_yaml {provider.task_yaml!r} does not exist")
    if provider.ssh_key and not Path(provider.ssh_key).exists():
        problems.append(f"sky provider: ssh_key {provider.ssh_key!r} does not exist")
    for env_name in provider.secrets.values():
        if not os.environ.get(env_name):
            problems.append(f"sky provider: secret variable {env_name} is not set")
    problems += aws_env_problems(provider, dict(os.environ))
    try:
        install_request_env_scope()  # local only: builds a request body, sends nothing
    except RuntimeError as error:
        problems.append(f"sky provider: {error}")
    return problems


def _uses_aws(provider: SkyProvider) -> bool:
    """Some offer may land on AWS: its infra / cloud is AWS, a wildcard (``*/<region>``),
    or not pinned at all."""
    for offer in provider.offers:
        where = str(offer.resources.get("infra") or offer.resources.get("cloud") or "").lower()
        if where.split("/", 1)[0] in ("", "*", "aws"):
            return True
    return False


def start_problems(cfg: dict[str, Any]) -> list[str]:
    """What refuses a sky pool's loop at its start (``pool --config``), before any order:
    a refusal at the first order would only cool that offer down and retry every tick."""
    try:
        provider = from_config(cfg, str(cfg.get("run") or cfg.get("shard_base")), sky=object())
    except (ValueError, TypeError, KeyError) as error:
        return [str(error)]
    problems = aws_env_problems(provider, dict(os.environ))
    try:
        install_request_env_scope()
    except RuntimeError as error:
        problems.append(f"sky provider: {error}")
    return problems


def start_refusal(cfg: dict[str, Any]) -> str | None:
    """The message a sky pool's start (``pool --config`` or an in-process controller) exits with."""
    refused = start_problems(cfg)
    return f"pool {cfg.get('shard_base')}: " + "; ".join(refused) if refused else None


def aws_env_problems(provider: SkyProvider, env: dict[str, str]) -> list[str]:
    """The run env's AWS settings that would reach sky's AWS calls, for a provider whose
    offers may land on AWS (Nebius / Kubernetes pools make no AWS call):

    * ``AWS_PROFILE`` / ``AWS_DEFAULT_PROFILE`` without ``provider.aws_profile``: the
      provider's requests drop them, but a local API server sky starts inherits them;
    * ``AWS_ACCESS_KEY_ID`` / ``_SECRET_ACCESS_KEY`` / ``_SESSION_TOKEN``, ``aws_profile``
      or not: in an inherited environment botocore prefers them to any profile;
    * ``AWS_ENDPOINT_URL[_STS|_EC2]``, ``AWS_SHARED_CREDENTIALS_FILE``, ``AWS_CONFIG_FILE``:
      never forwarded per request, but inherited the same way (an R2 endpoint would
      take sky's STS and EC2 calls)."""
    if not _uses_aws(provider):
        return []
    found = [n for n in AWS_KEY_ENVS + AWS_INHERITED_ENVS if env.get(n)]
    if not provider.aws_profile:
        found = [n for n in AWS_PROFILE_ENVS if env.get(n)] + found
    if not found:
        return []
    return [f"sky provider: the run env sets {', '.join(found)}, which sky would use for its own AWS calls "
            "(forwarded to a local sky API server, or inherited by one sky starts). Give the object store its "
            "credentials with YETO_RL_STORE_S3_PROFILE (and YETO_RL_STORE_S3_ENDPOINT_URL) instead of AWS_*, "
            "and name sky's AWS profile with provider.aws_profile"]
