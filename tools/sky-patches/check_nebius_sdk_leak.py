"""Replay the Nebius part of one status-refresh cycle, N times, in a local process.

Per cycle, like backend_utils.refresh_cluster_records() in SkyPilot 0.13.0:
  subprocess_utils.run_in_parallel over the given clusters (a ThreadPool with
  get_parallel_threads() workers); each worker calls
  Nebius.get_user_identities() (owner-identity check) and
  provision_lib.query_instances('Nebius', ...) (cloud status query);
  then annotations.clear_request_level_cache() as daemons.py does.

Read-only: reads handles and cluster YAML from ~/.sky/state.db, calls Nebius
GetProfile / ListProjects / ListInstances. No DB writes, no ssh, no cloud
mutations. Does not print credentials.

Usage: python -I check_nebius_sdk_leak.py <installed|patched> <cycles> <out.tsv> <cluster>...
  installed: run the adaptor that is installed in this venv.
  patched: exec $PATCHED_ADAPTOR into sky.adaptors.nebius first.
"""
import gc
import os
import sys
import threading
import time


def stats():
    st = {}
    with open('/proc/self/status', encoding='utf-8') as f:
        for line in f:
            k, _, v = line.partition(':')
            if k in ('VmRSS', 'Threads'):
                st[k] = int(v.split()[0])
    ep = 0
    for fd in os.listdir('/proc/self/fd'):
        try:
            if 'eventpoll' in os.readlink(f'/proc/self/fd/{fd}'):
                ep += 1
        except OSError:
            pass
    st['eventpoll'] = ep
    st['sdk_loops'] = sum(
        1 for t in threading.enumerate() if t.name == 'nebius-sdk-loop')
    return st


def main():
    mode, cycles, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    names = sys.argv[4:]
    from sky import clouds
    from sky import global_user_state
    from sky import provision as provision_lib
    from sky.adaptors import nebius
    from sky.utils import annotations
    from sky.utils import subprocess_utils
    if mode == 'patched':
        src = os.environ['PATCHED_ADAPTOR']
        exec(compile(open(src, encoding='utf-8').read(), src, 'exec'),
             nebius.__dict__)
    targets = []
    for n in names:
        h = global_user_state.get_handle_from_cluster_name(n)
        cfg = global_user_state.get_cluster_yaml_dict(h.cluster_yaml)
        targets.append((n, h.cluster_name_on_cloud, cfg['provider']))

    def one(t):
        n, on_cloud, provider = t
        clouds.Nebius.get_user_identities()
        return provision_lib.query_instances('Nebius', n, on_cloud, provider)

    with open(out, 'w', encoding='utf-8') as f:
        f.write('cycle\tt_s\tthreads\trss_kb\teventpoll\tsdk_loops\tpool_threads\terr\n')
        s = stats()
        f.write(f"0\t0\t{s['Threads']}\t{s['VmRSS']}\t{s['eventpoll']}\t{s['sdk_loops']}\t-\t\n")
        t0 = time.time()
        for c in range(1, cycles + 1):
            err = ''
            try:
                subprocess_utils.run_in_parallel(one, targets)
            except Exception as e:  # pylint: disable=broad-except
                err = repr(e)[:150].replace('\t', ' ').replace('\n', ' ')
            annotations.clear_request_level_cache()
            gc.collect()
            s = stats()
            f.write(f"{c}\t{time.time()-t0:.1f}\t{s['Threads']}\t{s['VmRSS']}\t"
                    f"{s['eventpoll']}\t{s['sdk_loops']}\t"
                    f"{subprocess_utils.get_parallel_threads()}\t{err}\n")
            f.flush()
    sys.stdout.flush()
    os._exit(0)


if __name__ == '__main__':
    main()
