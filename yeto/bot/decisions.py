"""Decision-request cards and their append-only JSONL audit log.

A decision request is a card with a title, evidence lines, a suggested option
and one button per option. Pressing a button only appends a record to the audit
log and replies that the operator will act: the bot executes nothing. Requests
are written to the same log, so a request posted by one process (for example
``python -m yeto.bot --request-decision``) is answered by the running bot.

Record types: ``{"type": "request", "id", "title", "evidence", "options",
"suggested", "ts"}`` and ``{"type": "decision", "id", "option", "user",
"user_id", "channel", "chat", "ts"}`` (``user`` is the display name, ``user_id``
the platform id the allowlist checks).
"""

from __future__ import annotations

import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, field

from .render import Message, b


@dataclass
class DecisionRequest:
    title: str
    evidence: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    suggested: str = ""
    id: str = ""
    ts: float = 0.0

    def message(self) -> Message:
        lines = [b("Evidence"), *[f"- {x}" for x in self.evidence]]
        if self.suggested:
            lines += ["", f"{b('Suggested')}: {self.suggested}"]
        lines += ["", "Pressing a button records the decision for the operator; the bot runs nothing."]
        buttons = [(opt + (" (suggested)" if opt == self.suggested else ""), f"/decide {self.id} {opt}")
                   for opt in self.options]
        return Message(f"decision needed: {self.title}", lines, "decision", buttons)


class DecisionLog:
    def __init__(self, path: str, *, clock=time.time):
        self.path = path
        self.clock = clock
        self.lock = threading.Lock()

    def _append(self, rec: dict) -> None:
        with self.lock:
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())

    def records(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
        return out

    def request(self, title: str, evidence: list[str], options: list[str], suggested: str = "") -> DecisionRequest:
        if not options or any(" " in o or not o for o in options):
            raise ValueError("decision options must be non-empty single words")
        if suggested and suggested not in options:
            raise ValueError(f"suggested option {suggested!r} is not one of {options}")
        req = DecisionRequest(title, list(evidence), list(options), suggested, "d" + secrets.token_hex(3),
                              self.clock())
        self._append({"type": "request", "id": req.id, "title": title, "evidence": req.evidence,
                      "options": req.options, "suggested": suggested, "ts": req.ts})
        return req

    def decide(self, req_id: str, option: str, *, user: str, channel: str, chat: str,
               user_id: str = "") -> Message:
        recs = self.records()
        req = next((r for r in recs if r.get("type") == "request" and r.get("id") == req_id), None)
        if req is None:
            return Message("unknown decision", [f"no decision request {req_id!r}"], "warn")
        if option not in (req.get("options") or []):
            return Message("unknown option", [f"{option!r} is not one of {req.get('options')}"], "warn")
        done = next((r for r in recs if r.get("type") == "decision" and r.get("id") == req_id), None)
        if done is not None:
            return Message("decision already recorded",
                           [f"{req.get('title')}: {done.get('option')} by {done.get('user') or '?'} at"
                            f" {time.strftime('%m-%d %H:%M', time.localtime(done.get('ts') or 0))}"], "info")
        self._append({"type": "decision", "id": req_id, "option": option, "user": user, "user_id": user_id,
                      "channel": channel, "chat": chat, "ts": self.clock()})
        return Message("decision recorded", [f"{req.get('title')}: {b(option)} (by {user or '?'})",
                                             "recorded; operator will act"], "ok")
