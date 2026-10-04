"""python -m yeto.sky_patches install|uninstall|status

`install` writes a persistent hook for this environment's sky (then
restart its API server: `sky api stop`); `uninstall` removes it."""

import sys
from pathlib import Path

from . import PTH_NAME, ensure_local_pth, pth_repo, uninstall


def main(argv=None) -> int:
    import os
    import sysconfig

    cmd = (argv or sys.argv[1:] or ["status"])[0]
    path = os.path.join(sysconfig.get_paths()["purelib"], PTH_NAME)
    if cmd == "install":
        repo = str(Path(__file__).resolve().parents[2])
        p, fresh = ensure_local_pth(repo)
        print(f"{'installed' if fresh else 'already installed'}: {p} -> {repo}; restart sky's API server: sky api stop")
    elif cmd == "uninstall":
        print(f"removed {uninstall()}" if pth_repo(path) else f"no yeto hook at {path}")
    elif cmd == "status":
        print(f"{path}: {pth_repo(path) or 'not installed'}")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
