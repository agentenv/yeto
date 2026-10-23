"""is_island0.py TAPE: exit 0 when the container tape has a heartbeat of island 0."""
import json
import sys


def main(path: str) -> int:
    with open(path, encoding="utf-8", errors="replace") as tape:
        for line in tape:
            try:
                event = json.loads(line)
            except ValueError:
                continue  # a line cut by the puller
            if event.get("event") == "rl_heartbeat" and event.get("island_id") == 0:
                return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
