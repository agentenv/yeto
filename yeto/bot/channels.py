"""Chat channels: Telegram (Bot API long polling), Lark app bot, Lark group webhook.

Every channel implements the same small interface:

``send(msg, chat=None) -> bool``
    post a :class:`~yeto.bot.render.Message` to ``chat`` or the report chat:
    buttons (command buttons only where the press comes back to the bot, URL
    buttons everywhere), charts (PNG when matplotlib and an upload path exist,
    else a text sparkline) and an @ of the owners when ``msg.mention``;
``poll_commands(timeout) -> list[Command]``
    block up to ``timeout`` seconds for incoming commands and button presses
    (empty for send-only channels);
``answer(cmd, text)``
    acknowledge a button press.

All network calls go through an injected ``http`` callable (see ``yeto.bot.http``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import queue
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Callable

from . import charts as charts_mod
from .http import HttpError, HttpFn, redact_url
from .render import BOLD_CLOSE, BOLD_OPEN, LEVEL_ICON, Message, is_url, strip_bold

TELEGRAM_MAX = 4000  # Bot API limit is 4096 characters
TELEGRAM_CAPTION_MAX = 1000
LARK_MAX = 8000  # keep cards well under the 30 KB request limit
LARK_TOKEN_EXPIRED = (99991661, 99991663, 99991668)
LARK_TEMPLATE = {"p0": "red", "p1": "orange", "p2": "grey", "warn": "orange", "ok": "green", "info": "blue",
                 "report": "indigo", "decision": "purple"}


@dataclass
class Command:
    channel: str
    chat: str
    text: str
    user: str = ""  # display handle for replies and logs only, never for authorization
    button: bool = False
    ref: Any = None  # channel-specific handle (Telegram callback query id)
    user_id: str = ""  # stable id the allowlist is checked against (Telegram from.id, Lark open_id)
    trusted: bool = False  # issued locally (``python -m yeto.bot --command``), not by a chat user


def log(text: str) -> None:
    print(f"[yeto-bot] {text}", file=sys.stderr, flush=True)


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 20].rstrip() + "\n... (truncated)"


def with_chart_text(msg: Message, failed: list) -> Message:
    """Inline sparklines for charts that could not be sent as images."""
    extra = [line for c in failed for line in charts_mod.text_lines(c)]
    return replace(msg, lines=msg.lines + ([""] + extra if extra else [])) if extra else msg


class Channel:
    name = "channel"
    can_receive = False
    chat_env = ""  # environment variables that configure the report chat / the allowlist (for messages)
    allow_env = ""

    def __init__(self, configured_chat: str = "", *, owners: list[str] | None = None,
                 allowed: list[str] | None = None):
        self.configured_chat = str(configured_chat or "")
        self.registered_chat = ""
        self.owners = list(owners or [])  # @ on P0
        self.allowed = list(allowed or [])  # user ids who may command / press; empty: anyone in the chat

    @property
    def report_chat(self) -> str:
        return self.configured_chat or self.registered_chat

    def user_allowed(self, cmd: Command) -> bool:
        """Only the platform's stable user id counts: display names are chosen by the
        sender (Telegram first names are free text, usernames can be released and
        taken by someone else)."""
        return not self.allowed or (bool(cmd.user_id) and cmd.user_id in self.allowed)

    def send(self, msg: Message, chat: str | None = None) -> bool:
        raise NotImplementedError

    def poll_commands(self, timeout: float = 25.0) -> list[Command]:
        time.sleep(timeout)
        return []

    def answer(self, cmd: Command, text: str) -> None:
        pass


# -- Telegram ---------------------------------------------------------------------

_TG_SPECIAL = re.compile(r"([_*`\[])")


def tg_markdown(msg: Message, owners: list[str] | None = None) -> str:
    """Legacy Telegram Markdown: bold spans become ``*x*`` (special characters inside
    are replaced, since entities cannot contain escapes); everything else is escaped
    with a backslash."""
    def conv(text: str) -> str:
        out, pos = [], 0
        for m in re.finditer(f"{BOLD_OPEN}(.*?){BOLD_CLOSE}", text, re.S):
            out.append(_TG_SPECIAL.sub(r"\\\1", text[pos:m.start()]))
            inner = m.group(1).replace("*", "x").replace("_", "-").replace("`", "'").replace("[", "(")
            out.append(f"*{inner}*")
            pos = m.end()
        out.append(_TG_SPECIAL.sub(r"\\\1", strip_bold(text[pos:])))
        return "".join(out)

    icon = LEVEL_ICON.get(msg.level, "")
    head = conv(f"{icon} {BOLD_OPEN}{msg.title}{BOLD_CLOSE}".strip())
    lines = [head, *(conv(line) for line in msg.lines)]
    if msg.mention and owners:
        lines.append(conv("cc " + " ".join(f"@{o}" for o in owners)))
    return truncate("\n".join(lines), TELEGRAM_MAX)


def tg_plain(msg: Message, owners: list[str] | None = None) -> str:
    icon = LEVEL_ICON.get(msg.level, "")
    text = f"{icon} {msg.plain()}".strip()
    if msg.mention and owners:
        text += "\ncc " + " ".join(f"@{o}" for o in owners)
    return truncate(text, TELEGRAM_MAX)


def tg_keyboard(msg: Message) -> dict | None:
    if not msg.buttons:
        return None
    row = [{"text": label, "url": target} if is_url(target) else {"text": label, "callback_data": target[:64]}
           for label, target in msg.buttons]
    return {"inline_keyboard": [row[i:i + 4] for i in range(0, len(row), 4)]}


def _tg_user(frm: dict | None) -> tuple[str, str]:
    """(display name, numeric user id) of a Telegram ``from`` object; only the id is trusted."""
    frm = frm or {}
    uid = str(frm.get("id") or "")
    return str(frm.get("username") or frm.get("first_name") or uid), uid


class TelegramChannel(Channel):
    name = "telegram"
    can_receive = True
    chat_env, allow_env = "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS"

    def __init__(self, token: str, chat: str = "", *, api_base: str = "https://api.telegram.org",
                 http: HttpFn, owners: list[str] | None = None, allowed: list[str] | None = None):
        super().__init__(chat, owners=owners, allowed=allowed)
        self.token = token
        self.api_base = api_base.rstrip("/")
        self.http = http
        self.offset = 0

    def api(self, method: str, files: dict | None = None, **params: Any) -> Any:
        url = f"{self.api_base}/bot{self.token}/{method}"
        if files:
            return self.http("POST", url, body={k: str(v) for k, v in params.items()}, files=files, timeout=60)
        return self.http("POST", url, body=params, timeout=35)

    def send(self, msg: Message, chat: str | None = None) -> bool:
        target = chat or self.report_chat
        if not target:
            return False
        pngs = [(c, charts_mod.render_png(c)) for c in msg.charts]
        msg = with_chart_text(msg, [c for c, png in pngs if png is None])
        extra = {}
        kb = tg_keyboard(msg)
        if kb:
            extra["reply_markup"] = kb
        ok = False
        try:
            self.api("sendMessage", chat_id=target, text=tg_markdown(msg, self.owners), parse_mode="Markdown", **extra)
            ok = True
        except Exception as exc:  # bad markup: retry as plain text
            log(f"telegram send failed, retrying as plain text: {exc}")
            try:
                self.api("sendMessage", chat_id=target, text=tg_plain(msg, self.owners), **extra)
                ok = True
            except Exception as exc2:
                log(f"telegram plain send failed: {exc2}")
        for chart, png in pngs:
            if png is None:
                continue
            try:
                self.api("sendPhoto", files={"photo": ("chart.png", png, "image/png")}, chat_id=target,
                         caption=chart.title[:TELEGRAM_CAPTION_MAX])
            except Exception as exc:
                log(f"telegram sendPhoto failed: {exc}")
        return ok

    def poll_commands(self, timeout: float = 25.0) -> list[Command]:
        try:
            res = self.api("getUpdates", offset=self.offset, timeout=int(timeout),
                           allowed_updates=["message", "callback_query"])
        except Exception as exc:
            log(f"telegram getUpdates failed: {exc}")
            time.sleep(5)
            return []
        return self.parse_updates(res)

    def parse_updates(self, res: Any) -> list[Command]:
        out = []
        for upd in (res or {}).get("result") or []:
            self.offset = max(self.offset, int(upd.get("update_id", 0)) + 1)
            cq = upd.get("callback_query")
            if cq:
                chat = ((cq.get("message") or {}).get("chat") or {}).get("id")
                if chat is not None:
                    name, uid = _tg_user(cq.get("from"))
                    out.append(Command(self.name, str(chat), cq.get("data") or "", name, True, cq.get("id"), uid))
                continue
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat = (msg.get("chat") or {}).get("id")
            if chat is None or not text:
                continue
            name, uid = _tg_user(msg.get("from"))
            out.append(Command(self.name, str(chat), text, name, user_id=uid))
        return out

    def answer(self, cmd: Command, text: str) -> None:
        if cmd.button and cmd.ref:
            try:
                self.api("answerCallbackQuery", callback_query_id=cmd.ref, text=text[:190])
            except Exception as exc:
                log(f"telegram answerCallbackQuery failed: {exc}")


# -- Lark ---------------------------------------------------------------------------

_LARK_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", "*": "&#42;", "_": "&#95;", "~": "&#126;",
             "`": "&#96;", "[": "&#91;", "]": "&#93;"}


def lark_md(text: str) -> str:
    esc = "".join(_LARK_ESC.get(ch, ch) for ch in text)
    return esc.replace(BOLD_OPEN, "**").replace(BOLD_CLOSE, "**")


def lark_card(msg: Message, *, buttons: bool, owners: list[str] | None = None,
              images: list[tuple[str, str]] | None = None) -> dict:
    """Interactive card (message card JSON 1.0): coloured header by level, one
    lark_md block, chart images, then buttons (command buttons only when
    ``buttons``; URL buttons always)."""
    icon = LEVEL_ICON.get(msg.level, "")
    body = truncate("\n".join(lark_md(line) for line in msg.lines), LARK_MAX) or " "
    if msg.mention and owners:
        body += "\n" + " ".join(f"<at id={o}></at>" for o in owners)
    elements: list[dict] = [{"tag": "div", "text": {"tag": "lark_md", "content": body}}]
    for title, key in images or []:
        elements.append({"tag": "img", "img_key": key, "alt": {"tag": "plain_text", "content": title}})
    actions = []
    for label, target in msg.buttons:
        if is_url(target):
            actions.append({"tag": "button", "text": {"tag": "plain_text", "content": label}, "type": "default",
                            "url": target})
        elif buttons:
            actions.append({"tag": "button", "text": {"tag": "plain_text", "content": label},
                            "type": "primary" if not actions else "default", "value": {"cmd": target}})
    if actions:
        elements.append({"tag": "action", "actions": actions})
    return {"config": {"wide_screen_mode": True},
            "header": {"template": LARK_TEMPLATE.get(msg.level, "blue"),
                       "title": {"tag": "plain_text", "content": f"{icon} {strip_bold(msg.title)}".strip()}},
            "elements": elements}


def lark_plain(msg: Message) -> str:
    return truncate(f"{LEVEL_ICON.get(msg.level, '')} {msg.plain()}".strip(), LARK_MAX)


def _lark_code(resp: Any) -> int | None:
    if not isinstance(resp, dict):
        return None
    code = resp.get("code", resp.get("StatusCode"))
    return int(code) if isinstance(code, (int, float)) else None


def _lark_call(fn: Callable[[], Any]) -> tuple[int | None, Any]:
    """Run one Lark request; HTTP errors carry the JSON body with the error code."""
    try:
        resp = fn()
    except HttpError as exc:
        try:
            resp = json.loads(exc.body)
        except ValueError:
            resp = {"code": -exc.status, "msg": exc.body[:200]}
    return _lark_code(resp), resp


class LarkAppChannel(Channel):
    """Self-built app bot: tenant_access_token + im/v1/messages (+ im/v1/images for
    charts); commands arrive through the long-connection client (``yeto.bot.lark_ws``)."""

    name = "lark"
    chat_env, allow_env = "LARK_CHAT_ID", "LARK_ALLOWED_OPEN_IDS"

    def __init__(self, app_id: str, app_secret: str, chat: str = "", *, domain: str = "https://open.larksuite.com",
                 http: HttpFn, refresh_margin: float = 600.0, clock: Callable[[], float] = time.time,
                 receive: bool = False, buttons: bool = False, owners: list[str] | None = None,
                 allowed: list[str] | None = None):
        super().__init__(chat, owners=owners, allowed=allowed)
        self.app_id, self.app_secret = app_id, app_secret
        self.domain = domain.rstrip("/")
        self.http = http
        self.refresh_margin = refresh_margin
        self.clock = clock
        self.can_receive = receive
        self.buttons = buttons and receive  # a press is only answered over the long connection
        self._token: str | None = None
        self._expires = 0.0
        self.inbox: queue.Queue = queue.Queue()
        self._seen: deque = deque(maxlen=2000)

    # token -------------------------------------------------------------------
    def token(self, force: bool = False) -> str:
        now = self.clock()
        if not force and self._token and now < self._expires - self.refresh_margin:
            return self._token
        code, resp = _lark_call(lambda: self.http(
            "POST", f"{self.domain}/open-apis/auth/v3/tenant_access_token/internal",
            body={"app_id": self.app_id, "app_secret": self.app_secret}, timeout=15))
        if code != 0 or not resp.get("tenant_access_token"):
            raise RuntimeError(f"lark tenant_access_token failed: code {code} {resp.get('msg') if resp else ''}")
        self._token = resp["tenant_access_token"]
        self._expires = now + float(resp.get("expire") or 7200)
        return self._token

    def _authed(self, fn: Callable[[str], Any]) -> tuple[int | None, Any]:
        """Call with a token; on an expired-token code refresh once and retry."""
        code, resp = _lark_call(lambda: fn(self.token()))
        if code in LARK_TOKEN_EXPIRED:
            code, resp = _lark_call(lambda: fn(self.token(force=True)))
        return code, resp

    def _post_message(self, chat: str, msg_type: str, content: dict) -> tuple[int | None, Any]:
        body = {"receive_id": chat, "msg_type": msg_type, "content": json.dumps(content, ensure_ascii=False)}
        url = f"{self.domain}/open-apis/im/v1/messages?receive_id_type=chat_id"
        return self._authed(lambda tok: self.http("POST", url, body=body,
                                                  headers={"Authorization": f"Bearer {tok}"}, timeout=15))

    def upload_image(self, png: bytes) -> str | None:
        """im/v1/images (scope im:resource) -> image_key, or None."""
        url = f"{self.domain}/open-apis/im/v1/images"
        code, resp = self._authed(lambda tok: self.http(
            "POST", url, body={"image_type": "message"}, files={"image": ("chart.png", png, "image/png")},
            headers={"Authorization": f"Bearer {tok}"}, timeout=30))
        key = ((resp or {}).get("data") or {}).get("image_key") if code == 0 else None
        if not key:
            log(f"lark image upload failed (code {code}: {(resp or {}).get('msg')}); using a text sparkline")
        return key

    def send(self, msg: Message, chat: str | None = None) -> bool:
        target = chat or self.report_chat
        if not target:
            return False
        try:
            images, failed = [], []
            for c in msg.charts:
                png = charts_mod.render_png(c)
                key = self.upload_image(png) if png else None
                if key:
                    images.append((c.title, key))
                else:
                    failed.append(c)
            msg = with_chart_text(msg, failed)
            card = lark_card(msg, buttons=self.buttons, owners=self.owners, images=images)
            code, resp = self._post_message(target, "interactive", card)
            if code == 0:
                return True
            log(f"lark card send failed (code {code}: {(resp or {}).get('msg')}), retrying as text")
            code, resp = self._post_message(target, "text", {"text": lark_plain(msg)})
            if code == 0:
                return True
            log(f"lark text send failed (code {code}: {(resp or {}).get('msg')})")
        except Exception as exc:
            log(f"lark send failed: {exc}")
        return False

    # commands (filled by the ws event client) -----------------------------------
    def push(self, cmd: Command, event_id: str = "") -> None:
        if event_id:
            if event_id in self._seen:  # Lark redelivers events it thinks were not acked
                return
            self._seen.append(event_id)
        self.inbox.put(cmd)

    def poll_commands(self, timeout: float = 25.0) -> list[Command]:
        out = []
        try:
            out.append(self.inbox.get(timeout=timeout))
            while True:
                out.append(self.inbox.get_nowait())
        except queue.Empty:
            pass
        return out


def webhook_sign(secret: str, timestamp: int) -> str:
    """Lark custom-bot signature: HMAC-SHA256 keyed by ``"{timestamp}\\n{secret}"``
    over an empty message, base64-encoded."""
    key = f"{timestamp}\n{secret}".encode("utf-8")
    return base64.b64encode(hmac.new(key, b"", hashlib.sha256).digest()).decode("ascii")


class LarkWebhookChannel(Channel):
    """Group custom bot: one-way push to a fixed group (no callbacks, no image upload)."""

    name = "lark-webhook"

    def __init__(self, url: str, secret: str = "", *, http: HttpFn, clock: Callable[[], float] = time.time,
                 owners: list[str] | None = None):
        super().__init__("webhook", owners=owners)
        self.url, self.secret = url, secret
        self.http = http
        self.clock = clock

    def _payload(self, extra: dict) -> dict:
        if not self.secret:
            return extra
        ts = int(self.clock())
        return {"timestamp": str(ts), "sign": webhook_sign(self.secret, ts), **extra}

    def send(self, msg: Message, chat: str | None = None) -> bool:
        msg = with_chart_text(msg, msg.charts)
        try:
            code, resp = _lark_call(lambda: self.http(
                "POST", self.url, body=self._payload({"msg_type": "interactive",
                                                      "card": lark_card(msg, buttons=False, owners=self.owners)}),
                timeout=15))
            if code == 0:
                return True
            log(f"lark webhook card failed (code {code}: {(resp or {}).get('msg')}), retrying as text")
            code, resp = _lark_call(lambda: self.http(
                "POST", self.url, body=self._payload({"msg_type": "text", "content": {"text": lark_plain(msg)}}),
                timeout=15))
            if code == 0:
                return True
            log(f"lark webhook text failed (code {code}: {(resp or {}).get('msg')})")
        except Exception as exc:
            log(f"lark webhook send failed: {exc}")
        return False


# -- dry run ------------------------------------------------------------------------

def dry_run_http(out: Callable[[str], None] = print) -> HttpFn:
    """An ``http`` stand-in that prints each request and answers like the real API."""
    def fake(method: str, url: str, *, body: Any = None, headers: dict | None = None, timeout: float = 0,
             files: dict | None = None) -> Any:
        shown = dict(body or {})
        if "app_secret" in shown:
            shown["app_secret"] = "<redacted>"
        for k, (fname, data, ctype) in (files or {}).items():
            shown[k] = f"<{ctype} {fname}, {len(data)} bytes>"
        safe_url = redact_url(url)
        if "getUpdates" in url:
            time.sleep(min(timeout, 1.0))
            return {"ok": True, "result": []}
        out(f"[dry-run] {method} {safe_url}\n{json.dumps(shown, ensure_ascii=False, indent=2)}")
        if "tenant_access_token" in url:
            return {"code": 0, "tenant_access_token": "dry-run", "expire": 7200}
        if "im/v1/images" in url:
            return {"code": 0, "data": {"image_key": "img_dry_run"}}
        return {"ok": True, "code": 0, "result": {}}
    return fake
