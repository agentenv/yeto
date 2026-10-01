"""Local head (cf. rl-engine-ports default-params-modal) with the syncer port moved to 29410."""
import json, os, shlex, sys
sys.argv = ["yeto"] + shlex.split(open(sys.argv[1]).read())
from yeto import cli, launcher, runs
from yeto.gpu_spec import parse_gpu_spec
launcher.SYNCER_PORT = 29410
args = cli.build_parser().parse_args(sys.argv[1:])
launcher.prepare_launch_args(args)
d = cli._serializable_args(args)
name = args.cluster_prefix
runs.create_run(name, d)
runs.update_run(name, controller="local", pid=os.getpid(),
                clusters=launcher.learner_cluster_names(name, parse_gpu_spec(args.gpu)))
print("[local-head] syncer port", launcher.SYNCER_PORT, flush=True)
sys.exit(cli.cmd_head(json.dumps(d)))
