"""``python -m yeto.node run --assignment <key>``: one sampler on this node.

The same steps on every source (an NC lease, a Daytona sandbox, a sky VM, an
own SSH node, a machine opened by hand), in order:

1. read the assignment ``assignments/<ticket>.json`` from the run store with the
   ticket's token (``YETO_NODE_TOKEN`` or ``--token-file``);
2. preflight (:mod:`.preflight`): the node's facts and checks for its profile;
3. the yeto tree by hash (:mod:`.code`): ``<work>/yeto/<sha>``;
4. the model by the profile's source (:mod:`.model`);
5. the spec's machine paths filled from the profile (:mod:`.paths`), written to
   ``<work>/sampler/<shard>.json``; a container launch without an image gets the
   engine image catalog's choice for (profile, model, engine) or is refused;
6. the sampler started as this process's child (:mod:`.launch`): a process, or a
   container in the foreground; SIGTERM / SIGINT are passed on (the sampler's
   orderly leave); its exit code becomes this process's.

Each step heartbeats ``nodes/<ticket>/g<generation>/heartbeat.json`` with its stage; the
sampler then writes ``ready`` at its JOIN and its own heartbeats; this process
writes ``exit.json`` with the exit code and the reason. A refused step writes
the exit record too (reason ``preflight`` / ``code`` / ``model`` / ``image``),
so a control plane that only reads the store knows why the node stopped.

``--dry-run`` reads the assignment and prints the plan (profile, paths, the
command, the environment's names) without running anything.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from yeto.launch.node_inventory import NodeProfile

from . import code as node_code
from . import launch as node_launch
from . import model as node_model
from . import paths as node_paths
from . import preflight as node_preflight
from .profile import ProfileError, is_container, node_dirs, profile_from_doc
from .records import TOKEN_ENV, Assignment, AssignmentError, NodeReporter, read_assignment, ticket_of_key
from .store import open_objects, read_env_file

# the runtime's own exit codes for a step that refused (the sampler's are its own)
EXIT_ASSIGNMENT = 7
EXIT_PREFLIGHT = 4
EXIT_CODE = 5
EXIT_MODEL = 6
EXIT_IMAGE = 8
EXIT_REASONS = {0: "finished", 3: "not_run", 2: "spec_rejected", 75: "reclaimed"}


class BootError(RuntimeError):
    def __init__(self, stage: str, code: int, message: str):
        super().__init__(message)
        self.stage, self.code = stage, code


@dataclass
class Plan:
    """What the runtime starts: the sampler's command, its environment (non-secret),
    where its spec and log live on this host, and the facts of the preflight."""

    argv: list[str]
    env: dict[str, str]
    spec_host_path: str
    log_host_path: str
    yeto_dir: str
    spec: dict[str, Any] = field(default_factory=dict)
    image: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {"argv": self.argv, "env": sorted(self.env), "spec": self.spec_host_path, "log": self.log_host_path,
                "yeto": self.yeto_dir, "image": self.image, "model": self.model}


class NodeRuntime:
    def __init__(self, objects: Any, assignment: Assignment, profile: NodeProfile, *, store_uri: str | None,
                 environ: Mapping[str, str] | None = None, env_file: str | None = None,
                 heartbeat_s: float | None = None, run: Callable[..., Any] = subprocess.run,
                 popen: Callable[..., Any] = subprocess.Popen, log: Callable[[str], None] | None = None,
                 skip_preflight: bool = False) -> None:
        self.objects, self.assignment, self.profile = objects, assignment, profile
        self.store_uri, self.env_file, self.heartbeat_s = store_uri, env_file, heartbeat_s
        self.environ = dict(os.environ if environ is None else environ)
        self.secrets = read_env_file(env_file) if env_file else {}
        self.run_, self.popen, self.skip_preflight = run, popen, skip_preflight
        self.log = log or (lambda line: print(f"[yeto-node] {line}", flush=True))
        self.reporter = NodeReporter(objects, assignment.ticket, assignment.generation,
                                     base={"sampler": assignment.shard, "source": assignment.source})
        self.child: Any = None
        self._stopping = False

    # -- steps
    def _stage(self, stage: str, **fields: Any) -> None:
        self.log(f"{self.assignment.ticket}: {stage}")
        self.reporter.heartbeat(stage, **fields)

    def preflight(self) -> dict[str, Any]:
        req = node_preflight.Requirements.from_doc(self.assignment.preflight)
        if self.assignment.sudo:
            req.sudo = True
        if self.skip_preflight:
            return {}
        try:
            found, checks = node_preflight.run_preflight(self.profile, req, cards=self.assignment.cards,
                                                         run=self.run_, environ=self.environ)
        except node_preflight.PreflightError as error:
            raise BootError("preflight", EXIT_PREFLIGHT, str(error)) from None
        warn = [c.name for c in checks if not c.ok]
        return {**found.record(), **({"warnings": warn} if warn else {})}

    def code(self) -> str:
        try:
            return node_code.ensure_code(self.objects, self.assignment.code, node_dirs(self.profile)["work"])
        except (node_code.CodeError, OSError) as error:
            raise BootError("code", EXIT_CODE, f"code {self.assignment.code.sha}: {error}") from None

    def model(self) -> dict[str, Any]:
        try:
            return node_model.prepare_model(self.assignment.model, models_dir=node_dirs(self.profile)["models"],
                                            objects=self.objects, environ={**self.environ, **self.secrets},
                                            run=self.run_, log=self.log)
        except (node_model.ModelError, OSError, subprocess.TimeoutExpired) as error:
            raise BootError("model", EXIT_MODEL, str(error)) from None

    def image(self, spec: Mapping[str, Any]) -> tuple[str | None, str | None]:
        """``(image, python)`` of a container launch (the catalog's choice when the
        assignment and the profile name none); ``(None, None)`` for a process."""
        if not is_container(self.profile):
            return None, self.assignment.python
        from yeto.launch import engine_images  # noqa: PLC0415

        named = self.assignment.image or self.profile.image
        needs = engine_images.spec_needs(spec)
        model, pref = needs.pop("model"), needs.pop("engine_pref")
        try:
            chosen = engine_images.resolve(self.profile, model, pref, image=named, **needs)
        except engine_images.ImageIncompatible as error:
            raise BootError("image", EXIT_IMAGE, str(error)) from None
        return chosen.ref, self.assignment.python or chosen.python or "python3"

    def plan(self, yeto_dir: str, facts: Mapping[str, Any] | None = None) -> Plan:
        a, p = self.assignment, self.profile
        spec = node_paths.fill_spec(a.spec, p, run_prefix=a.run)
        if a.continues_shard and not spec.get("continues_shard"):
            spec["continues_shard"] = a.continues_shard
        image, python = self.image(spec)
        dirs = node_paths.sampler_dirs(p)
        sampler_yeto = node_code.code_dir(dirs["work"], a.code.sha)
        paths = node_launch.sampler_paths(p, a.shard)
        host_paths = (facts or {}).get("host_paths")
        env = node_launch.sampler_env(a, p, yeto_dir=sampler_yeto, store_uri=self.store_uri or a.store_uri,
                                      environ={**self.environ, **self.secrets}, heartbeat_s=self.heartbeat_s,
                                      host_paths=host_paths)
        if is_container(p):
            assert image is not None
            argv = node_launch.container_argv(p, name=node_launch.container_name(a.shard), image=image, env=env,
                                              python=str(python), spec_path=paths["spec"],
                                              env_files=[self.env_file] if self.env_file else [], cards=a.cards,
                                              host_paths=host_paths, sudo=a.sudo)
        else:
            argv = node_launch.process_argv(python or sys.executable, paths["spec"])
        return Plan(argv=argv, env=env, spec_host_path=node_paths.host_path(paths["spec"], p),
                    log_host_path=node_paths.host_path(paths["log"], p), yeto_dir=yeto_dir, spec=spec, image=image,
                    facts=dict(facts or {}))

    def write_spec(self, plan: Plan) -> None:
        p, spec = self.profile, plan.spec
        path = Path(plan.spec_host_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(spec, indent=1))
        tmp.replace(path)
        for key in ("event_tape",):  # the sampler's own directories on this host
            value = spec.get(key)
            if isinstance(value, str) and value.startswith("/"):
                Path(node_paths.host_path(value, p)).parent.mkdir(parents=True, exist_ok=True)
        Path(node_paths.host_path(f"{node_paths.sampler_dirs(p)['work']}/stage", p)).mkdir(parents=True, exist_ok=True)

    def boot(self) -> Plan:
        self._stage("preflight")
        facts = self.preflight()
        self._stage("code", facts=facts)
        yeto_dir = self.code()
        self._stage("model", yeto=yeto_dir)
        model = self.model()
        self._stage("spec", model=model)
        plan = self.plan(yeto_dir, facts)
        plan.model = model
        self.write_spec(plan)
        return plan

    # -- the sampler
    def _forward(self, signum: int, _frame: Any) -> None:
        self._stopping = True
        child = self.child
        if child is None:
            return
        if is_container(self.profile):
            self.run_(node_launch.stop_argv(self.profile, node_launch.container_name(self.assignment.shard),
                                            sudo=self.assignment.sudo), capture_output=True, timeout=60, check=False)
        else:
            try:
                child.send_signal(signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

    def start(self, plan: Plan) -> int:
        child_env = {**self.environ, **self.secrets, **plan.env} if not is_container(self.profile) \
            else dict(self.environ)
        log = Path(plan.log_host_path)
        log.parent.mkdir(parents=True, exist_ok=True)
        self._stage("starting", argv0=plan.argv[0], image=plan.image)
        handlers = {}
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                handlers[sig] = signal.signal(sig, self._forward)
            except ValueError:  # not the main thread (tests): no forwarding
                pass
        try:
            with log.open("ab") as out:
                self.child = self.popen(plan.argv, env=child_env, stdout=out, stderr=subprocess.STDOUT,
                                        cwd=node_dirs(self.profile)["work"], start_new_session=False)
                code = self.child.wait()
        finally:
            for sig, old in handlers.items():
                signal.signal(sig, old)
        return int(code)

    def exit_reason(self, code: int) -> str:
        if code in EXIT_REASONS:
            return EXIT_REASONS[code]
        if code < 0:
            return f"signal {-code}"
        return "stopped" if self._stopping else "error"

    def main(self, *, dry_run: bool = False) -> int:
        if dry_run:
            plan = self.plan(node_code.code_dir(node_dirs(self.profile)["work"], self.assignment.code.sha))
            print(json.dumps({"ticket": self.assignment.ticket, "shard": self.assignment.shard,
                              "generation": self.assignment.generation, "profile": self.profile.summary(),
                              **plan.record()}, indent=1, sort_keys=True))
            return 0
        try:
            plan = self.boot()
        except BootError as error:
            self.log(f"{self.assignment.ticket}: {error.stage} refused: {error}")
            self.reporter.exit(error.code, error.stage, detail=str(error)[:2000])
            return error.code
        except Exception as error:  # noqa: BLE001 - written to the store, then raised
            self.reporter.exit(1, "boot", detail=f"{type(error).__name__}: {error}"[:2000])
            raise
        code = self.start(plan)
        self.log(f"{self.assignment.ticket}: sampler exit {code}")
        self.reporter.exit(code, self.exit_reason(code))
        return code


def load_profile_override(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    text = Path(path).expanduser().read_text()
    if path.endswith((".yaml", ".yml")):
        import yaml  # noqa: PLC0415

        doc = yaml.safe_load(text)
    else:
        doc = json.loads(text)
    if not isinstance(doc, dict):
        raise ProfileError(f"{path}: a node profile must be a mapping")
    return doc


def read_token(token_file: str | None, environ: Mapping[str, str]) -> str:
    if token_file:
        return Path(token_file).expanduser().read_text().strip()
    return str(environ.get(TOKEN_ENV) or "").strip()


def run_cli(args: Any, *, environ: Mapping[str, str] | None = None, run: Callable[..., Any] = subprocess.run,
            popen: Callable[..., Any] = subprocess.Popen) -> int:
    env = dict(os.environ if environ is None else environ)
    store = args.store or env.get("YETO_NODE_STORE")
    if not store:
        print("[yeto-node] --store (or YETO_NODE_STORE) is required: the run store URI or directory", file=sys.stderr)
        return 2
    objects = open_objects(store, env_file=args.env_file, environ=env)
    ticket = ticket_of_key(args.assignment)
    try:
        assignment = read_assignment(objects, ticket, read_token(args.token_file, env))
    except AssignmentError as error:
        print(f"[yeto-node] {error}", file=sys.stderr)
        return EXIT_ASSIGNMENT
    override = load_profile_override(args.profile)
    if args.work or args.models or args.cache:
        paths = dict(override.get("paths") or (assignment.profile or {}).get("paths") or {})
        paths.update({k: v for k, v in (("work", args.work), ("models", args.models), ("cache", args.cache)) if v})
        override["paths"] = paths
    if not assignment.profile and not override:
        print("[yeto-node] the assignment has no node profile: pass --profile", file=sys.stderr)
        return 2
    profile = profile_from_doc({**dict(assignment.profile or {}), **override})
    runtime = NodeRuntime(objects, assignment, profile, store_uri=store if "://" in store else None, environ=env,
                          env_file=args.env_file, heartbeat_s=args.heartbeat_s, run=run, popen=popen,
                          skip_preflight=args.skip_preflight)
    return runtime.main(dry_run=args.dry_run)


def add_run_args(parser: Any) -> None:
    parser.add_argument("--assignment", required=True, help="the ticket, or assignments/<ticket>.json")
    parser.add_argument("--store", help="the run store: s3://BUCKET/runs/PREFIX, r2://..., or a directory "
                                        "(default: $YETO_NODE_STORE)")
    parser.add_argument("--env-file", help="the store env file (AWS_*, R2_BUCKET): read here, passed to the sampler")
    parser.add_argument("--token-file", help=f"a file holding the ticket's token (default: ${TOKEN_ENV})")
    parser.add_argument("--profile", help="a JSON / YAML node profile: fields that replace the assignment's")
    parser.add_argument("--work", help="this node's work directory (profile paths.work)")
    parser.add_argument("--models", help="this node's model directory (profile paths.models)")
    parser.add_argument("--cache", help="this node's cache directory (profile paths.cache)")
    parser.add_argument("--heartbeat-s", type=float, default=None, help="the sampler's heartbeat period")
    parser.add_argument("--skip-preflight", action="store_true", help="start without the node preflight")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")

