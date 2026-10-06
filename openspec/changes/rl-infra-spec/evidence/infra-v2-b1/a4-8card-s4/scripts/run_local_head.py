"""Local head controller: same code path as `yeto launch` head mode (cli._head ->
launcher.run(local_syncer=...)), but the 'head' is this machine (public IP),
because no ports-capable non-Nebius cloud is enabled for sky here."""
import json, os, shlex, sys
sys.argv = ["yeto"] + shlex.split(open(sys.argv[1]).read())
from yeto import cli, launcher, runs
from yeto.gpu_spec import parse_gpu_spec
args = cli.build_parser().parse_args(sys.argv[1:])
launcher.prepare_launch_args(args)
d = cli._serializable_args(args)
name = args.cluster_prefix
runs.create_run(name, d)
runs.update_run(name, controller="local", pid=os.getpid(),
    clusters=launcher.learner_cluster_names(name, parse_gpu_spec(args.gpu)))
print("[local-head] rl_engine arg =", d.get("rl_engine"), " rl_image =", d.get("rl_image"), flush=True)
sys.exit(cli.cmd_head(json.dumps(d)))
