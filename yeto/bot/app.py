"""Bot core: command routing with chat / user authorization, the alert monitor,
the scheduled progress card and daily digest, decision requests.

Phase-2 hook: text that is not a slash command goes to ``Bot.nl`` (a callable
``(text, ReadOnlyTools) -> Message``) when one is installed, e.g. an LLM that
answers questions with the read-only dashboard tools; without it such text is
ignored.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import Counter
from dataclasses import replace
from typing import Callable

from . import lark_ws
from .channels import Channel, Command, LarkAppChannel, LarkWebhookChannel, TelegramChannel, dry_run_http, log
from .config import Config
from .decisions import DecisionLog
from .http import DashboardClient, HttpFn, request_json
from .monitor import Monitor
from .render import (Message, b, ckpt_message, details_message, digest_message, eval_message, fdur, help_message,
                     island_message, merge_message, res_message, status_message)
from .rules import P0, P1, P2, SEV_NAME, hbm_limit, hbm_warn, rss_warn

VIEW_COMMANDS = ("/status", "/island", "/eval", "/merge", "/res", "/ckpt", "/details")
LOCAL_COMMANDS = ("/help", "/start", "/alerts", "/ack", "/mute", "/unmute", "/decide")
# commands (and their buttons) that change state other people rely on: refused
# while the channel has no user allowlist, since then anyone in the report chat
# could run them. Value: why, for the refusal message.
ALERT_CONTROL_COMMANDS = {"/ack": "changes alerting for everyone", "/mute": "changes alerting for everyone",
                          "/unmute": "changes alerting for everyone",
                          "/decide": "records a decision the operator will act on"}
_DUR = re.compile(r"^(\d+(?:\.\d+)?)([smhd]?)$")


def parse_command(text: str) -> tuple[str, str]:
    """``"/Island@yeto_bot 2"`` -> ``("/island", "2")``."""
    head, _, rest = text.strip().partition(" ")
    return head.split("@", 1)[0].lower(), rest.strip()


def parse_duration(text: str) -> float | None:
    m = _DUR.match(text.strip().lower())
    if not m:
        return None
    return float(m.group(1)) * {"": 60, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


class ReadOnlyTools:
    """What a phase-2 natural-language handler may call: dashboard reads and alert state, nothing else."""

    def __init__(self, client: DashboardClient, monitor: Monitor):
        self._client, self._monitor = client, monitor

    def view(self) -> dict:
        return self._client.view()

    def bad_rounds(self) -> list[dict]:
        return self._client.bad_rounds()

    def active_alerts(self) -> list[dict]:
        return [{"key": list(k), "sev": SEV_NAME[c.sev], "title": c.title, "lines": c.lines}
                for k, c in self._monitor.active.items()]


class Bot:
    def __init__(self, cfg: Config, client: DashboardClient, channels: list[Channel], *,
                 clock: Callable[[], float] = time.time,
                 nl: Callable[[str, ReadOnlyTools], Message] | None = None):
        self.cfg = cfg
        self.client = client
        self.channels = channels
        self.clock = clock
        self.nl = nl
        self.monitor = Monitor(cfg.th, cfg.policy, clock=clock)
        self.decisions = DecisionLog(cfg.audit_log, clock=clock)
        self.last_progress = float("-inf")
        self.digest_day: tuple | None = None
        self.lock = threading.RLock()
        self._load_state()

    # -- registered chats (optional state file) -------------------------------------
    def _load_state(self) -> None:
        if not self.cfg.state_file or not os.path.exists(self.cfg.state_file):
            return
        try:
            with open(self.cfg.state_file) as fh:
                chats = json.load(fh).get("chats") or {}
        except (OSError, ValueError) as exc:
            log(f"state file unreadable, ignoring: {exc}")
            return
        for ch in self.channels:
            if not ch.configured_chat and chats.get(ch.name):
                ch.registered_chat = str(chats[ch.name])

    def _save_state(self) -> None:
        if not self.cfg.state_file:
            return
        chats = {ch.name: ch.registered_chat for ch in self.channels if ch.registered_chat}
        tmp = self.cfg.state_file + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"chats": chats}, fh)
        os.replace(tmp, self.cfg.state_file)

    # -- commands -------------------------------------------------------------------
    def alerts_line(self) -> str:
        c = self.monitor.active_counts()
        line = f"{b('Alerts')} active P0 {c[P0]} | P1 {c[P1]}"
        now = self.clock()
        if now < self.monitor.mute_until:
            line += f" | muted for {fdur(self.monitor.mute_until - now)}"
        return line

    def progress(self, view: dict) -> Message:
        return status_message(view, alerts_line=self.alerts_line())

    def reply_for(self, name: str, arg: str, cmd: Command) -> Message:
        if name in ("/help", "/start"):
            return help_message()
        if name == "/alerts":
            return self.monitor.alerts_message()
        if name == "/ack":
            return self._ack(arg, cmd)
        if name == "/mute":
            parts = arg.split()
            secs = parse_duration(parts[0]) if parts else None
            if secs is None:
                return Message("usage", ["/mute <30m|2h|1d> [all]  (all: P0 too)"], "warn")
            everything = len(parts) > 1 and parts[1].lower() == "all"
            self.monitor.mute(secs, include_p0=everything)
            log(f"muted for {secs:.0f}s by {cmd.user or '?'} (P0 {'too' if everything else 'still pages'})")
            return Message("muted", [f"{'all alerts' if everything else 'P1/P2'} muted for {fdur(secs)}"
                                     f" by {cmd.user or '?'}" + ("" if everything else "; P0 still pages")], "ok")
        if name == "/unmute":
            self.monitor.unmute()
            return Message("unmuted", [f"alerts resumed by {cmd.user or '?'}"], "ok")
        if name == "/decide":
            req_id, _, option = arg.partition(" ")
            return self.decisions.decide(req_id, option.strip(), user=cmd.user, channel=cmd.channel, chat=cmd.chat,
                                         user_id=cmd.user_id)
        if name not in VIEW_COMMANDS:
            return help_message(f"unknown command {name}")
        view = self.client.view()
        th = self.cfg.th
        if name == "/status":
            return self.progress(view)
        if name == "/island":
            return island_message(view, arg, hbm_limit=hbm_limit(view, th))
        if name == "/eval":
            return eval_message(view)
        if name == "/merge":
            if not view.get("sft") and not view.get("rounds_syncer"):
                try:
                    view = dict(view, rounds_syncer=self.client.bad_rounds())
                except Exception as exc:
                    log(f"/api/rounds failed: {exc}")
            return merge_message(view)
        if name == "/res":
            return res_message(view, hbm_limit=hbm_limit(view, th), hbm_warn=hbm_warn(view, th),
                               rss_warn=rss_warn(view, th), host_min=th.host_avail_min_bytes)
        if name == "/ckpt":
            return ckpt_message(view)
        notices = self.monitor.card_notices(arg)  # /details cN
        if notices is None:
            return Message("unknown card", [f"no alert card {arg!r} (cards are kept in memory)"], "warn")
        return details_message(view, notices, ref=arg)

    def _ack(self, arg: str, cmd: Command) -> Message:
        ref = arg.strip().lower()
        if ref in ("", "all"):
            keys = self.monitor.ack(None, cmd.user)
            titles = [self.monitor.active[k].title for k in keys]
        else:
            notices = self.monitor.card_notices(ref)
            if notices is None:
                return Message("unknown card", [f"no alert card {arg!r}"], "warn")
            live = [n for n in notices if n.kind != "resolve"]
            keys = self.monitor.ack([n.key for n in live], cmd.user)
            titles = list(dict.fromkeys(n.title for n in live))
        if not keys:
            return Message("nothing to ack", ["no active alert subjects"], "info")
        log(f"acked {len(keys)} subject(s) by {cmd.user or '?'}: {titles}")
        return Message(f"acked {len(keys)} alert(s) for {fdur(self.cfg.policy.ack_mute_s)}",
                       [f"by {cmd.user or '?'}", *titles], "ok")

    def send(self, ch: Channel, msg: Message, chat: str | None = None) -> bool:
        if self.cfg.title_prefix:
            msg = replace(msg, title=self.cfg.title_prefix + msg.title)
        return ch.send(msg, chat)

    def _register(self, ch: Channel, cmd: Command) -> Message | None:
        """``/start`` without a configured report chat: the chat becomes the report chat,
        but only for an allowlisted user. Without an allowlist anyone who can reach the
        bot could claim the run's alerts, so registration is refused."""
        who = f"chat {cmd.chat} user {cmd.user or '?'} ({cmd.user_id or 'no id'})"
        if not ch.allowed:
            log(f"{ch.name}: refusing /start from {who}: configure the report chat id or a user allowlist")
            reply = Message("registration disabled", [
                "This bot has no report chat and no user allowlist, so it does not let a chat register itself.",
                f"Set {ch.chat_env or 'the report chat id'} or {ch.allow_env or 'a user allowlist'} and restart."],
                "warn")
            ch.answer(cmd, "Not allowed")
            self.send(ch, reply, cmd.chat)
            return reply
        if not ch.user_allowed(cmd):
            log(f"{ch.name}: refusing /start from {who}: not in the allowlist")
            ch.answer(cmd, "Not allowed")
            return None
        ch.registered_chat = cmd.chat
        log(f"{ch.name}: report chat registered: {cmd.chat} (by {cmd.user or '?'} {cmd.user_id})")
        self._save_state()
        reply = help_message("Registered this chat for training alerts.")
        ch.answer(cmd, "ok")
        self.send(ch, reply, cmd.chat)
        return reply

    def handle(self, ch: Channel, cmd: Command) -> Message | None:
        """Authorize and answer one command; returns the reply that was sent (or None)."""
        slash = cmd.text.strip().startswith("/")
        name, arg = parse_command(cmd.text) if slash else ("", cmd.text.strip())
        target = ch.report_chat
        if not target:
            if not slash:
                return None
            if name == "/start":
                return self._register(ch, cmd)
            reply = Message("not registered", ["send /start in the chat that should receive alerts"], "info")
            ch.answer(cmd, "ok")
            self.send(ch, reply, cmd.chat)
            return reply
        # run data stays in the report chat; local CLI commands skip the user allowlist
        if str(cmd.chat) != str(target) or not (cmd.trusted or ch.user_allowed(cmd)):
            log(f"{ch.name}: ignoring {name or 'text'} from chat {cmd.chat} user {cmd.user or '?'}")
            ch.answer(cmd, "Not allowed")
            return None
        if not slash:
            if self.nl is None:
                return None
            try:
                reply = self.nl(arg, ReadOnlyTools(self.client, self.monitor))
            except Exception as exc:
                reply = Message("question failed", [str(exc)[:500]], "warn")
            self.send(ch, reply, cmd.chat)
            return reply
        if name in ALERT_CONTROL_COMMANDS and not ch.allowed and not cmd.trusted:
            log(f"{ch.name}: refusing {name} from user {cmd.user or '?'} ({cmd.user_id or 'no id'}):"
                f" no user allowlist configured")
            ch.answer(cmd, "Needs a user allowlist")
            reply = Message("not allowed", [
                f"{name} {ALERT_CONTROL_COMMANDS[name]}, so it needs a user allowlist"
                f" ({ch.allow_env or 'the channel allowlist'}); read-only commands still work."], "warn")
            self.send(ch, reply, cmd.chat)
            return reply
        ch.answer(cmd, f"running {name}")
        try:
            with self.lock:  # monitor state is shared with the monitor thread
                reply = self.reply_for(name, arg, cmd)
        except Exception as exc:
            reply = Message("command failed", [f"{name}: {exc}"[:500]], "warn")
        self.send(ch, reply, cmd.chat)
        return reply

    # -- broadcasts -----------------------------------------------------------------
    def broadcast(self, msgs: list[Message]) -> None:
        for msg in msgs:
            for ch in self.channels:
                if ch.report_chat:
                    self.send(ch, msg)

    def request_decision(self, title: str, evidence: list[str], options: list[str], suggested: str = "") -> Message:
        """Post a decision card to every report chat; presses are only recorded."""
        msg = self.decisions.request(title, evidence, options, suggested).message()
        self.broadcast([msg])
        return msg

    def _digest_due(self, now: float) -> bool:
        at = self.cfg.policy.digest_at
        if at is None:
            return False
        t = time.localtime(now)
        day, minute = (t.tm_year, t.tm_yday), t.tm_hour * 60 + t.tm_min
        if self.digest_day is None:  # started after today's slot: first digest tomorrow
            self.digest_day = day if minute >= at else (t.tm_year, t.tm_yday - 1)
        if minute >= at and day != self.digest_day:
            self.digest_day = day
            return True
        return False

    def digest(self, view: dict) -> Message:
        hist = [h for h in self.monitor.history_since(self.clock() - 86400) if h[5] and h[4] != "resolve"]
        counts = {SEV_NAME[s]: sum(1 for h in hist if h[1] == s) for s in (P0, P1, P2)}
        top = [f"{n}x {t}" for t, n in Counter(h[3] for h in hist if h[1] <= P1).most_common(5)]
        return digest_message(view, alert_counts=counts, top_alerts=top)

    def tick(self) -> list[Message]:
        """One monitor pass (+ the progress card / digest when due); returns what was broadcast."""
        with self.lock:
            msgs = self._tick()
        self.broadcast(msgs)
        return msgs

    def _tick(self) -> list[Message]:
        try:
            view = self.client.view()
        except Exception as exc:
            log(f"dashboard fetch failed: {exc}")
            card = self.monitor.card(self.monitor.fetch_failed(exc), link=self.cfg.dashboard_link)
            return [card] if card else []
        card = self.monitor.card(self.monitor.evaluate(view), link=self.cfg.dashboard_link)
        msgs = [card] if card else []
        now = self.clock()
        every = self.cfg.policy.progress_interval
        if every > 0 and now - self.last_progress >= every:
            self.last_progress = now
            msgs.append(self.progress(view))
        if self._digest_due(now):
            msgs.append(self.digest(view))
        return msgs

    # -- threads --------------------------------------------------------------------
    def _monitor_loop(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                log(f"monitor pass failed: {exc}")
            stop.wait(self.cfg.poll_interval)

    def _poll_loop(self, ch: Channel, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                cmds = ch.poll_commands(25)
            except Exception as exc:
                log(f"{ch.name}: poll failed: {exc}")
                stop.wait(5)
                continue
            for cmd in cmds:
                try:
                    self.handle(ch, cmd)
                except Exception as exc:
                    log(f"{ch.name}: command {cmd.text!r} failed: {exc}")

    def allowlist_warnings(self) -> list[str]:
        return [f"{ch.name}: no user allowlist ({ch.allow_env or 'allowlist'} unset): anyone in the report chat can"
                f" run read-only commands; {', '.join(ALERT_CONTROL_COMMANDS)} are refused"
                for ch in self.channels if ch.can_receive and not ch.allowed]

    def run(self, stop: threading.Event | None = None) -> None:
        stop = stop or threading.Event()
        for line in self.allowlist_warnings():
            log("warning: " + line)
        threads = [threading.Thread(target=self._monitor_loop, args=(stop,), name="monitor", daemon=True)]
        threads += [threading.Thread(target=self._poll_loop, args=(ch, stop), name=f"poll-{ch.name}", daemon=True)
                    for ch in self.channels if ch.can_receive]
        for t in threads:
            t.start()
        try:
            while not stop.is_set():
                stop.wait(3600)
        except KeyboardInterrupt:
            stop.set()


def build_channels(cfg: Config, http: HttpFn | None = None, *, receive: bool = True) -> list[Channel]:
    """Every channel whose configuration is complete; the rest are skipped with one log line.
    ``receive=False`` (one-shot CLI runs, dry run) never opens a long connection or polls."""
    http = http or (dry_run_http() if cfg.dry_run else request_json)
    receive = receive and not cfg.dry_run
    out: list[Channel] = []
    if cfg.telegram_token:
        out.append(TelegramChannel(cfg.telegram_token, cfg.telegram_chat_id, api_base=cfg.telegram_api_base,
                                   http=http, owners=cfg.telegram_owners, allowed=cfg.telegram_allowed))
        out[-1].can_receive = receive
    else:
        log("telegram: TELEGRAM_BOT_TOKEN not set, skipping")
    lark = build_lark(cfg, http, receive)
    if lark:
        out.append(lark)
    return out


def build_lark(cfg: Config, http: HttpFn, receive: bool = True) -> Channel | None:
    has_app = bool(cfg.lark_app_id and cfg.lark_app_secret)
    hook = (LarkWebhookChannel(cfg.lark_webhook_url, cfg.lark_webhook_secret, http=http, owners=cfg.lark_owners)
            if cfg.lark_webhook_url else None)
    mode = cfg.lark_mode
    if mode == "webhook" or (mode == "auto" and not has_app):
        if hook is None:
            log("lark: neither LARK_APP_ID/LARK_APP_SECRET nor LARK_WEBHOOK_URL set, skipping")
        else:
            log("lark: group webhook (send-only)" + (" with signature" if cfg.lark_webhook_secret else ""))
        return hook
    if not has_app:
        log(f"lark: LARK_MODE={mode} needs LARK_APP_ID and LARK_APP_SECRET, skipping")
        return None
    want_ws = mode in ("auto", "ws") and receive
    ws_ok = want_ws and lark_ws.available()
    if want_ws and not ws_ok:
        log("lark: lark-oapi not installed (pip install lark-oapi): no command receiving")
        if mode == "auto" and not cfg.lark_chat_id and hook is not None:
            log("lark: no LARK_CHAT_ID for the send-only app bot, falling back to the group webhook")
            return hook
    ch = LarkAppChannel(cfg.lark_app_id, cfg.lark_app_secret, cfg.lark_chat_id, domain=cfg.lark_domain, http=http,
                        refresh_margin=cfg.lark_token_refresh_margin, receive=ws_ok,
                        buttons=cfg.lark_card_buttons, owners=cfg.lark_owners, allowed=cfg.lark_allowed)
    if cfg.dry_run:  # render the cards as the listening bot would
        ch.buttons = cfg.lark_card_buttons
    if ws_ok:
        lark_ws.start(ch)
        log("lark: app bot with long-connection events")
    else:
        log("lark: app bot, send-only" + ("" if cfg.lark_chat_id else " (set LARK_CHAT_ID to receive alerts)"))
    return ch
