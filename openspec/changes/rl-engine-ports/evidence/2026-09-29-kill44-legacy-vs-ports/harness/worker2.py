import sys, os, importlib.util
MR = os.environ["KILL_MR"]
sys.path[:0] = [MR, "/work/yeto", "/work/harness"]
spec = importlib.util.spec_from_file_location("bench", "/work/yeto/scripts/benchmark_rl.py")
B = importlib.util.module_from_spec(spec); sys.modules["bench"] = B; spec.loader.exec_module(B)
from pathlib import Path
sys.exit(B.run_training_worker(Path(sys.argv[1])))
