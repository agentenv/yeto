"""``python -m yeto.bot``: run the bot, or do one thing and exit."""

from __future__ import annotations

import argparse
import json
import sys
import time

from .app import Bot, build_channels
from .channels import Command, log
from .config import Config
from .http import DashboardClient, request_json


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m yeto.bot", description=__doc__)
    one = ap.add_mutually_exclusive_group()
    one.add_argument("--once", action="store_true",
                     help="run one monitor pass (sends the progress card) and exit")
    one.add_argument("--command", metavar="TEXT",
                     help="answer one command (e.g. '/status') into each channel's report chat and exit")
    one.add_argument("--request-decision", metavar="JSON",
                     help='post a decision card and exit: {"title", "evidence": [...], "options": [...], '
                          '"suggested"}; the running bot records the press in the audit log')
    args = ap.parse_args(argv)
    cfg = Config.from_env()
    oneshot = bool(args.once or args.command or args.request_decision)
    channels = build_channels(cfg, receive=not oneshot)
    client = DashboardClient(cfg.dashboard_url, timeout=cfg.dashboard_timeout, http=request_json)
    bot = Bot(cfg, client, channels)
    if args.command:
        for ch in channels:
            if ch.report_chat:
                # whoever can run this already holds the bot's credentials: trust it
                bot.handle(ch, Command(ch.name, ch.report_chat, args.command, "cli", trusted=True))
            else:
                log(f"{ch.name}: no report chat configured, skipping")
        return 0
    if args.request_decision:
        spec = json.loads(args.request_decision)
        bot.request_decision(spec["title"], spec.get("evidence") or [], spec["options"], spec.get("suggested", ""))
        return 0
    if args.once:
        bot.tick()
        return 0
    if not channels:
        log("no channel configured (TELEGRAM_BOT_TOKEN / LARK_APP_ID+LARK_APP_SECRET / LARK_WEBHOOK_URL): idling")
        while True:
            time.sleep(3600)
    log(f"started: dashboard {cfg.dashboard_url}, poll every {cfg.poll_interval:g}s, progress every "
        f"{cfg.policy.progress_interval:g}s, channels "
        + ", ".join(f"{c.name}({'rx' if c.can_receive else 'tx'}, chat {c.report_chat or 'unset'})" for c in channels)
        + (" [dry run]" if cfg.dry_run else ""))
    bot.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
