"""Lark long-connection (WebSocket) event client, an optional dependency.

The official SDK (``pip install lark-oapi``) opens an outbound WebSocket to the
Lark open platform, so the bot receives ``im.message.receive_v1`` (commands) and
``card.action.trigger`` (button presses) without a public HTTP endpoint. Without
the SDK the Lark app channel is send-only, or the bot falls back to the group
webhook (``LARK_WEBHOOK_URL``).

Event parsing lives in plain functions over attribute objects so it is testable
without the SDK.
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

from .channels import Command, LarkAppChannel, log

_MENTION = re.compile(r"@_user_\d+\s*")
_BACKOFF_START = 5.0
_BACKOFF_CAP = 300.0
_STABLE_S = 60.0  # a connection that lived this long resets the backoff


def available() -> bool:
    try:
        import lark_oapi  # noqa: F401
    except ImportError:
        return False
    return True


def _get(obj: Any, *path: str) -> Any:
    for p in path:
        if obj is None:
            return None
        obj = obj.get(p) if isinstance(obj, dict) else getattr(obj, p, None)
    return obj


def parse_message_event(data: Any) -> tuple[Command, str] | None:
    """``P2ImMessageReceiveV1`` -> (command, message id); None for non-text messages.
    Text without a leading ``/`` is kept for the natural-language hook."""
    msg = _get(data, "event", "message")
    if _get(msg, "message_type") != "text":
        return None
    try:
        text = json.loads(_get(msg, "content") or "{}").get("text") or ""
    except ValueError:
        return None
    text = _MENTION.sub("", text).strip()  # group messages carry "@_user_1 /status"
    chat = _get(msg, "chat_id")
    if not chat or not text:
        return None
    user = str(_get(data, "event", "sender", "sender_id", "open_id") or "")
    return Command("lark", str(chat), text, user, user_id=user), str(_get(msg, "message_id") or "")


def parse_card_event(data: Any) -> tuple[Command, str] | None:
    """``P2CardActionTrigger`` -> (command, event id) for buttons built by ``lark_card``."""
    value = _get(data, "event", "action", "value") or {}
    cmd = value.get("cmd") if isinstance(value, dict) else None
    chat = _get(data, "event", "context", "open_chat_id")
    if not cmd or not chat:
        return None
    user = str(_get(data, "event", "operator", "open_id") or "")
    event_id = _get(data, "header", "event_id") or ""
    return Command("lark", str(chat), str(cmd), user, button=True, user_id=user), str(event_id)


def card_toast(parsed: tuple[Command, str] | None) -> str:
    """Toast shown on the press itself. Neutral on purpose: the allowlist is checked
    later in ``Bot.handle``, whose reply says what ran or why it was refused."""
    return "received" if parsed else "unknown button"


def callbacks(channel: LarkAppChannel) -> tuple[Callable[[Any], None], Callable[[Any], Any]]:
    """(message handler, card-action handler) that feed ``channel``'s inbox."""

    def on_message(data: Any) -> None:
        parsed = parse_message_event(data)
        if parsed:
            channel.push(*parsed)

    def on_card(data: Any) -> Any:
        from lark_oapi.event.callback.model.p2_card_action_trigger import P2CardActionTriggerResponse

        parsed = parse_card_event(data)
        if parsed:
            channel.push(*parsed)
        return P2CardActionTriggerResponse({"toast": {"type": "info", "content": card_toast(parsed)}})

    return on_message, on_card


def build_handler(channel: LarkAppChannel) -> Any:
    import lark_oapi as lark

    on_message, on_card = callbacks(channel)
    builder = lark.EventDispatcherHandler.builder("", "").register_p2_im_message_receive_v1(on_message)
    if hasattr(builder, "register_p2_card_action_trigger"):
        builder = builder.register_p2_card_action_trigger(on_card)
    else:
        log("lark-oapi too old for card.action.trigger: buttons will not respond (upgrade lark-oapi)")
    return builder.build()


def start(channel: LarkAppChannel, *, stop: threading.Event | None = None,
          sleep: Callable[[float], None] = time.sleep,
          clock: Callable[[], float] = time.monotonic) -> threading.Thread:
    """Run the SDK's blocking ws client in a daemon thread, reconnecting with backoff.

    The backoff doubles while attempts keep failing fast and goes back to the base
    delay once a connection stayed up for ``_STABLE_S``, so a drop after a long
    healthy stretch reconnects quickly instead of waiting out the cap."""

    def run() -> None:
        import asyncio

        import lark_oapi as lark
        import lark_oapi.ws.client as ws_client

        # the SDK binds a module-level event loop at import; give this thread its own
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        ws_client.loop = loop
        backoff = _BACKOFF_START
        while stop is None or not stop.is_set():
            began = clock()
            try:
                cli = lark.ws.Client(channel.app_id, channel.app_secret, event_handler=build_handler(channel),
                                     domain=channel.domain, log_level=lark.LogLevel.WARNING)
                log("lark long connection starting")
                cli.start()
            except Exception as exc:
                log(f"lark long connection failed: {exc}; retrying in {backoff:.0f}s")
            if clock() - began >= _STABLE_S:
                backoff = _BACKOFF_START
            sleep(backoff)
            backoff = min(_BACKOFF_CAP, backoff * 2)

    t = threading.Thread(target=run, name="lark-ws", daemon=True)
    t.start()
    return t
