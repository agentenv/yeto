"""yeto.bot channels: Telegram / Lark message bodies, token refresh, webhook signature, fallbacks,
and an end-to-end pass against a real loopback dashboard server."""

import base64
import hashlib
import hmac
import json
import sys
import threading
from types import SimpleNamespace

import pytest

from yeto.bot import app as app_mod
from yeto.bot import channels as channels_mod
from yeto.bot import charts as charts_mod
from yeto.bot import lark_ws
from yeto.bot.charts import Chart, sparkline, text_lines
from yeto.bot.app import Bot, build_channels
from yeto.bot.channels import (LarkAppChannel, LarkWebhookChannel, TelegramChannel, lark_card,
                               tg_markdown, webhook_sign)
from yeto.bot.config import Config
from yeto.bot.http import DashboardClient, HttpError, multipart, redact_url, request_json
from yeto.bot.render import Message, b

from bot_helpers import FakeClock, FakeHttp

MSG = Message("isl_0 is stale", [f"{b('step')} 12 | loss 0.7 *x* [a]_b", "plain"], "p0",
              [("Status", "/status"), ("Island 0", "/island 0")])
CHART = Chart("loss", [("isl0", [1, 2, 3], [0.9, 0.8, 0.7])])


@pytest.fixture(autouse=True)
def no_matplotlib(monkeypatch):
    """Deterministic chart path: text sparklines unless a test installs a fake renderer."""
    monkeypatch.setattr(charts_mod, "available", lambda: False)


def fake_png(monkeypatch, png=b"\x89PNG fake"):
    monkeypatch.setattr(channels_mod.charts_mod, "render_png", lambda chart: png)


# -- Telegram -----------------------------------------------------------------------

def test_telegram_markdown_escapes_and_bold():
    text = tg_markdown(MSG)
    assert text.splitlines()[0] == "\U0001f6a8 *isl-0 is stale*"  # no escapes possible inside bold
    assert text.splitlines()[1] == r"*step* 12 | loss 0.7 \*x\* \[a]\_b"
    assert "\x02" not in text


def test_telegram_send_body_and_plain_fallback():
    http = FakeHttp([("sendMessage", [HttpError(400, "can't parse entities", "u"), {"ok": True}])])
    ch = TelegramChannel("123:abc", "-100", http=http)
    assert ch.send(MSG)
    first, second = http.bodies("sendMessage")
    assert http.calls[0]["url"] == "https://api.telegram.org/bot123:abc/sendMessage"
    assert first["chat_id"] == "-100" and first["parse_mode"] == "Markdown"
    assert first["reply_markup"] == {"inline_keyboard": [[{"text": "Status", "callback_data": "/status"},
                                                          {"text": "Island 0", "callback_data": "/island 0"}]]}
    assert "parse_mode" not in second and "*" not in second["text"].splitlines()[0]
    assert second["text"].startswith("\U0001f6a8 isl_0 is stale")


def test_telegram_send_without_chat_is_skipped_and_long_text_truncated():
    http = FakeHttp()
    ch = TelegramChannel("t", "", http=http)
    assert not ch.send(MSG) and http.calls == []
    ch.configured_chat = "1"
    ch.send(Message("big", ["x" * 5000]))
    text = http.bodies("sendMessage")[0]["text"]
    assert len(text) <= 4000 and text.endswith("(truncated)")


def test_telegram_parse_updates_messages_and_buttons():
    ch = TelegramChannel("t", "1", http=FakeHttp())
    cmds = ch.parse_updates({"result": [
        {"update_id": 10, "message": {"text": "/status", "chat": {"id": 1}, "from": {"username": "al"}}},
        {"update_id": 11, "message": {"text": "hello", "chat": {"id": 1}}},
        {"update_id": 12, "callback_query": {"id": "cq9", "data": "/island 0", "from": {"id": 77, "first_name": "Bo"},
                                             "message": {"chat": {"id": 1}, "message_id": 5}}}]})
    assert [(c.chat, c.text, c.user, c.user_id, c.button) for c in cmds] == [
        ("1", "/status", "al", "", False), ("1", "hello", "", "", False), ("1", "/island 0", "Bo", "77", True)]
    assert ch.offset == 13
    ch.answer(cmds[2], "running /island")
    assert ch.http.bodies("answerCallbackQuery") == [{"callback_query_id": "cq9", "text": "running /island"}]


def test_telegram_allowlist_checks_numeric_id_only():
    ch = TelegramChannel("t", "1", http=FakeHttp(), allowed=["42"])

    def press(frm):
        return ch.parse_updates({"result": [{"update_id": 1, "callback_query": {
            "id": "cq", "data": "/mute 1d all", "from": frm, "message": {"chat": {"id": 1}}}}]})[0]

    assert ch.user_allowed(press({"id": 42, "first_name": "Eve"}))
    assert not ch.user_allowed(press({"id": 7, "first_name": "42"}))  # a first name is free text
    assert not ch.user_allowed(press({"id": 7, "username": "42"}))
    assert not ch.user_allowed(press({"first_name": "42"}))  # no id at all


def test_telegram_poll_uses_offset_long_poll():
    http = FakeHttp([("getUpdates", {"ok": True, "result": [{"update_id": 4, "message": {
        "text": "/eval", "chat": {"id": 7}}}]})])
    ch = TelegramChannel("t", "7", http=http)
    assert [c.text for c in ch.poll_commands(25)] == ["/eval"]
    ch.poll_commands(25)
    bodies = http.bodies("getUpdates")
    assert bodies[0]["offset"] == 0 and bodies[1]["offset"] == 5 and bodies[0]["timeout"] == 25
    assert bodies[0]["allowed_updates"] == ["message", "callback_query"]


# -- Lark card --------------------------------------------------------------------------

def test_lark_card_structure_and_escaping():
    card = lark_card(MSG, buttons=True)
    assert card["header"]["template"] == "red"
    assert card["header"]["title"] == {"tag": "plain_text", "content": "\U0001f6a8 isl_0 is stale"}
    md = card["elements"][0]["text"]
    assert md["tag"] == "lark_md"
    assert md["content"].splitlines()[0] == "**step** 12 | loss 0.7 &#42;x&#42; &#91;a&#93;&#95;b"
    actions = card["elements"][1]["actions"]
    assert [a["value"] for a in actions] == [{"cmd": "/status"}, {"cmd": "/island 0"}]
    assert actions[0]["type"] == "primary"
    assert len(lark_card(MSG, buttons=False)["elements"]) == 1


# -- Lark app bot ---------------------------------------------------------------------

def token_resp(tok, expire=7200):
    return {"code": 0, "msg": "ok", "tenant_access_token": tok, "expire": expire}


def test_lark_app_send_body_and_token_cache_refresh():
    clock = FakeClock(1000.0)
    http = FakeHttp([("tenant_access_token", [token_resp("t-1"), token_resp("t-2")]),
                     ("im/v1/messages", {"code": 0, "data": {}})])
    ch = LarkAppChannel("cli_a", "sec", "oc_1", http=http, clock=clock, refresh_margin=600, receive=True,
                        buttons=True)
    assert ch.send(MSG)
    tok_call = http.calls[0]
    assert tok_call["url"] == "https://open.larksuite.com/open-apis/auth/v3/tenant_access_token/internal"
    assert tok_call["body"] == {"app_id": "cli_a", "app_secret": "sec"}
    send = http.calls[1]
    assert send["url"].endswith("/open-apis/im/v1/messages?receive_id_type=chat_id")
    assert send["headers"]["Authorization"] == "Bearer t-1"
    assert send["body"]["receive_id"] == "oc_1" and send["body"]["msg_type"] == "interactive"
    card = json.loads(send["body"]["content"])  # content is a JSON string
    assert card["elements"][1]["tag"] == "action"
    clock.t += 7200 - 601  # still valid outside the margin: cached
    ch.send(MSG)
    assert len(http.bodies("tenant_access_token")) == 1
    clock.t += 2  # inside the refresh margin: refreshed before expiry
    ch.send(MSG)
    assert len(http.bodies("tenant_access_token")) == 2
    assert http.calls[-1]["headers"]["Authorization"] == "Bearer t-2"


def test_lark_app_expired_token_retries_once_and_text_fallback():
    http = FakeHttp([("tenant_access_token", [token_resp("old"), token_resp("new")]),
                     ("im/v1/messages", [{"code": 99991663, "msg": "token expired"}, {"code": 0}])])
    ch = LarkAppChannel("a", "s", "oc", http=http)
    assert ch.send(MSG)
    sends = [c for c in http.calls if "messages" in c["url"]]
    assert [c["headers"]["Authorization"] for c in sends] == ["Bearer old", "Bearer new"]
    http2 = FakeHttp([("tenant_access_token", token_resp("t")),
                      ("im/v1/messages", [HttpError(400, json.dumps({"code": 230099, "msg": "bad card"}), "u"),
                                          {"code": 0}])])
    ch2 = LarkAppChannel("a", "s", "oc", http=http2)
    assert ch2.send(MSG)
    last = http2.bodies("messages")[-1]
    assert last["msg_type"] == "text" and json.loads(last["content"])["text"].startswith("\U0001f6a8 isl_0 is stale")


def test_lark_app_token_failure_and_no_chat():
    http = FakeHttp([("tenant_access_token", {"code": 10003, "msg": "invalid app_secret"})])
    ch = LarkAppChannel("a", "s", "oc", http=http)
    assert not ch.send(MSG)
    assert not LarkAppChannel("a", "s", "", http=http).send(MSG)


def test_lark_buttons_need_receive_and_opt_in():
    for receive, buttons, tags in ((False, True, ["div"]), (True, False, ["div"]), (True, True, ["div", "action"])):
        http = FakeHttp([("tenant_access_token", token_resp("t"))])
        LarkAppChannel("a", "s", "oc", http=http, receive=receive, buttons=buttons).send(MSG)
        card = json.loads(http.bodies("messages")[0]["content"])
        assert [e["tag"] for e in card["elements"]] == tags


# -- Lark webhook -------------------------------------------------------------------

def test_webhook_signature_matches_reference_algorithm():
    ts, secret = 1_700_000_000, "demo-secret"
    expect = base64.b64encode(hmac.new(f"{ts}\n{secret}".encode(), b"", hashlib.sha256).digest()).decode()
    assert webhook_sign(secret, ts) == expect


def test_webhook_payload_signed_and_fallback():
    http = FakeHttp([("hook", [{"code": 19021, "msg": "sign match fail"}, {"code": 0}])])
    ch = LarkWebhookChannel("https://open.feishu.cn/open-apis/bot/v2/hook/x", "sec", http=http,
                            clock=FakeClock(1_700_000_000.9))
    assert ch.report_chat and ch.send(MSG)
    card_body, text_body = http.bodies("hook")
    assert card_body["timestamp"] == "1700000000" and card_body["sign"] == webhook_sign("sec", 1_700_000_000)
    assert card_body["msg_type"] == "interactive" and isinstance(card_body["card"], dict)
    assert len(card_body["card"]["elements"]) == 1  # custom bots cannot take button callbacks
    assert text_body["msg_type"] == "text" and text_body["content"]["text"].startswith("\U0001f6a8")
    plain = FakeHttp([("hook", {"StatusCode": 0})])
    LarkWebhookChannel("https://x/hook/y", http=plain).send(MSG)
    assert "sign" not in plain.bodies("hook")[0]


# -- Lark long-connection events ------------------------------------------------------

def ns(**kw):
    return SimpleNamespace(**kw)


def test_lark_ws_message_and_card_parsing_with_dedupe():
    data = ns(event=ns(message=ns(message_type="text", content=json.dumps({"text": "@_user_1 /island 2"}),
                                  chat_id="oc_9", message_id="om_1"),
                       sender=ns(sender_id=ns(open_id="ou_7"))))
    cmd, mid = lark_ws.parse_message_event(data)
    assert (cmd.chat, cmd.text, cmd.user, cmd.button, mid) == ("oc_9", "/island 2", "ou_7", False, "om_1")
    data.event.message.content = json.dumps({"text": "@_user_1 how is island 2?"})
    assert lark_ws.parse_message_event(data)[0].text == "how is island 2?"  # kept for the NL hook
    data.event.message.message_type = "image"
    assert lark_ws.parse_message_event(data) is None
    card = ns(header=ns(event_id="ev1"), event=ns(action=ns(value={"cmd": "/status"}),
                                                  context=ns(open_chat_id="oc_9"), operator=ns(open_id="ou_7")))
    cmd2, eid = lark_ws.parse_card_event(card)
    assert cmd2.button and cmd2.text == "/status" and eid == "ev1"
    assert lark_ws.card_toast((cmd2, eid)) == "received" and lark_ws.card_toast(None) == "unknown button"
    ch = LarkAppChannel("a", "s", "oc_9", http=FakeHttp(), receive=True)
    ch.push(cmd, mid)
    ch.push(cmd, mid)  # redelivered
    ch.push(cmd2, eid)
    assert [c.text for c in ch.poll_commands(0.01)] == ["/island 2", "/status"]
    assert ch.poll_commands(0.01) == []


def test_lark_ws_callbacks_with_sdk():
    pytest.importorskip("lark_oapi")
    ch = LarkAppChannel("a", "s", "oc_9", http=FakeHttp(), receive=True)
    assert lark_ws.build_handler(ch) is not None
    on_message, on_card = lark_ws.callbacks(ch)
    on_message(ns(event=ns(message=ns(message_type="text", content=json.dumps({"text": "/eval"}), chat_id="oc_9",
                                      message_id="om_2"), sender=ns(sender_id=ns(open_id="ou_1")))))
    press = ns(header=ns(event_id="ev2"), event=ns(action=ns(value={"cmd": "/ack c1"}),
                                                   context=ns(open_chat_id="oc_9"), operator=ns(open_id="ou_1")))
    resp = on_card(press)
    assert resp.toast.content == "received"  # the press is not authorized yet
    assert [(c.text, c.button) for c in ch.poll_commands(0.01)] == [("/eval", False), ("/ack c1", True)]


# -- channel selection --------------------------------------------------------------

def test_build_channels_matrix(monkeypatch):
    monkeypatch.setattr(lark_ws, "available", lambda: False)
    http = FakeHttp()
    assert build_channels(Config(), http) == []
    chans = build_channels(Config(telegram_token="t", lark_webhook_url="https://x/hook"), http)
    assert [c.name for c in chans] == ["telegram", "lark-webhook"]
    # app creds without the SDK and without a chat id: fall back to the webhook
    chans = build_channels(Config(lark_app_id="a", lark_app_secret="s", lark_webhook_url="https://x/hook"), http)
    assert [c.name for c in chans] == ["lark-webhook"]
    chans = build_channels(Config(lark_app_id="a", lark_app_secret="s", lark_chat_id="oc"), http)
    assert chans[0].name == "lark" and not chans[0].can_receive
    assert build_channels(Config(lark_mode="ws", lark_webhook_url="https://x/hook"), http) == []
    started = []
    monkeypatch.setattr(lark_ws, "available", lambda: True)
    monkeypatch.setattr(lark_ws, "start", lambda ch: started.append(ch))
    chans = build_channels(Config(lark_app_id="a", lark_app_secret="s"), http)
    assert chans[0].can_receive and chans[0].buttons and started == chans
    chans = build_channels(Config(lark_app_id="a", lark_app_secret="s", lark_card_buttons=False), http)
    assert not chans[0].buttons
    started.clear()
    chans = build_channels(Config(lark_app_id="a", lark_app_secret="s", telegram_token="t"), http, receive=False)
    assert started == [] and not any(c.can_receive for c in chans)  # one-shot CLI runs never listen
    started.clear()
    chans = build_channels(Config(lark_app_id="a", lark_app_secret="s", dry_run=True), http)
    assert not chans[0].can_receive and started == []  # dry run never opens the connection


def test_dry_run_prints_bodies_and_redacts(capsys, monkeypatch):
    monkeypatch.setattr(lark_ws, "available", lambda: False)
    cfg = Config(dry_run=True, telegram_token="123:secret", telegram_chat_id="-1", lark_app_id="cli",
                 lark_app_secret="topsecret", lark_chat_id="oc_1")
    chans = build_channels(cfg)
    for ch in chans:
        assert ch.send(MSG)
    out = capsys.readouterr().out
    assert "POST https://api.telegram.org/bot<token>/sendMessage" in out
    assert "123:secret" not in out and "topsecret" not in out and "<redacted>" in out
    assert "/open-apis/im/v1/messages?receive_id_type=chat_id" in out
    hook = build_channels(Config(dry_run=True, lark_webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/"
                                                                 "a1b2-c3d4"))
    assert hook[0].send(MSG)
    out = capsys.readouterr().out
    assert "POST https://open.feishu.cn/open-apis/bot/v2/hook/<token>" in out and "a1b2-c3d4" not in out


def test_redact_url_hides_path_tokens():
    assert redact_url("https://api.telegram.org/bot123:AA-b_c/sendMessage") == \
        "https://api.telegram.org/bot<token>/sendMessage"
    assert redact_url("https://open.larksuite.com/open-apis/bot/v2/hook/0f9e-77aa") == \
        "https://open.larksuite.com/open-apis/bot/v2/hook/<token>"
    assert redact_url("https://open.feishu.cn/open-apis/bot/v2/hook/0f9e?x=1") == \
        "https://open.feishu.cn/open-apis/bot/v2/hook/<token>?x=1"
    plain = "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=chat_id"
    assert redact_url(plain) == plain


@pytest.mark.parametrize("url,secret", [("https://api.telegram.org/bot123:secret/sendMessage", "123:secret"),
                                        ("https://open.feishu.cn/open-apis/bot/v2/hook/hook-secret", "hook-secret")])
def test_http_error_never_carries_the_token(monkeypatch, url, secret):
    import io
    import urllib.error

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"code": 19001}'))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(HttpError) as info:
        request_json("POST", url, body={})
    assert secret not in str(info.value) and "<token>" in str(info.value)


# -- end to end: real loopback dashboard -> bot -> fake channels ----------------------

@pytest.fixture
def dashboard(tmp_path):
    from test_dashboard_sft import make_run

    from yeto.dashboard.reducer import Reducer
    from yeto.dashboard.serve import DashboardState, make_server

    root = make_run(tmp_path / "run", steps=8)
    state = DashboardState(Reducer(run="sft-e2e"), [str(root)])
    state.poll()
    srv = make_server(state, "127.0.0.1", 0, live=False)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_end_to_end_against_dashboard_server(dashboard, tmp_path):
    from bot_helpers import FakeChannel

    client = DashboardClient(dashboard, http=request_json)
    tg, lark = FakeChannel("telegram", "1"), FakeChannel("lark", "oc")
    bot = Bot(Config(dashboard_url=dashboard, audit_log=str(tmp_path / "audit.jsonl")), client, [tg, lark],
              clock=FakeClock())
    sent = bot.tick()
    assert [m.title for m in sent] == ["yeto progress: sft-e2e"]
    text = sent[0].plain()
    assert "8/100" in text and "Outer 2/960" in text and "Merge direct-in-round" in text
    assert [m.title for _, m in tg.sent] == [m.title for _, m in lark.sent]
    assert bot.handle(tg, app_mod.Command("telegram", "1", "/island 0")).title.startswith("island 0")
    merge = bot.handle(tg, app_mod.Command("telegram", "1", "/merge")).plain()
    assert "Timed-out rounds (1)" in merge and "rejected 1" in merge
    assert "Syncer checkpoint writes (1)" in bot.handle(tg, app_mod.Command("telegram", "1", "/ckpt")).plain()


# -- mentions, URL buttons, charts ------------------------------------------------------

ALERT = Message("P0 isl0 stale", ["no new step"], "p0",
                [("Ack 2h", "/ack c1"), ("Open dashboard", "https://d.example/")],
                [CHART], mention=True)


def test_telegram_mention_url_button_and_sparkline_fallback():
    http = FakeHttp()
    TelegramChannel("t", "1", http=http, owners=["ops_lead"]).send(ALERT)
    [body] = http.bodies("sendMessage")
    assert body["text"].splitlines()[-1] == r"cc @ops\_lead"
    assert "loss isl0: " in body["text"] and "0.7..0.9 (last 0.7)" in body["text"]
    assert body["reply_markup"]["inline_keyboard"] == [[{"text": "Ack 2h", "callback_data": "/ack c1"},
                                                        {"text": "Open dashboard", "url": "https://d.example/"}]]
    assert http.bodies("sendPhoto") == []


def test_telegram_send_photo_when_png_renders(monkeypatch):
    fake_png(monkeypatch)
    http = FakeHttp()
    TelegramChannel("t", "1", http=http).send(ALERT)
    [photo] = [c for c in http.calls if "sendPhoto" in c["url"]]
    assert photo["body"] == {"chat_id": "1", "caption": "loss"}
    assert photo["files"]["photo"] == ("chart.png", b"\x89PNG fake", "image/png")
    assert "loss isl0" not in http.bodies("sendMessage")[0]["text"]  # no sparkline when the image went out


def test_lark_image_upload_mention_and_fallback(monkeypatch):
    fake_png(monkeypatch)
    http = FakeHttp([("tenant_access_token", token_resp("t")),
                     ("im/v1/images", {"code": 0, "data": {"image_key": "img_v3_1"}})])
    ch = LarkAppChannel("a", "s", "oc", http=http, receive=True, buttons=True, owners=["ou_1"])
    assert ch.send(ALERT)
    up = next(c for c in http.calls if "im/v1/images" in c["url"])
    assert up["body"] == {"image_type": "message"} and up["files"]["image"][1] == b"\x89PNG fake"
    assert up["headers"]["Authorization"] == "Bearer t"
    card = json.loads(http.bodies("messages")[0]["content"])
    assert "<at id=ou_1></at>" in card["elements"][0]["text"]["content"]
    assert card["elements"][1] == {"tag": "img", "img_key": "img_v3_1", "alt": {"tag": "plain_text", "content": "loss"}}
    acts = card["elements"][2]["actions"]
    assert acts[0]["value"] == {"cmd": "/ack c1"} and acts[1]["url"] == "https://d.example/"
    bad = FakeHttp([("tenant_access_token", token_resp("t")),
                    ("im/v1/images", {"code": 99991672, "msg": "no im:resource"})])
    LarkAppChannel("a", "s", "oc", http=bad, receive=True, buttons=False).send(ALERT)
    card = json.loads(bad.bodies("messages")[0]["content"])
    assert [e["tag"] for e in card["elements"]] == ["div", "action"]  # URL button stays, command button dropped
    assert "loss isl0: " in card["elements"][0]["text"]["content"]


def test_webhook_uses_sparklines_and_keeps_url_buttons():
    http = FakeHttp([("hook", {"code": 0})])
    LarkWebhookChannel("https://x/hook/y", http=http, owners=["ou_1"]).send(ALERT)
    card = http.bodies("hook")[0]["card"]
    body = card["elements"][0]["text"]["content"]
    assert "loss isl0: " in body and "<at id=ou_1></at>" in body
    assert card["elements"][1]["actions"] == [
        {"tag": "button", "text": {"tag": "plain_text", "content": "Open dashboard"}, "type": "default",
         "url": "https://d.example/"}]


def test_sparkline_and_text_lines():
    assert sparkline([1, 2, 3, 4, 5, 6, 7, 8]) == "\u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    assert sparkline([2, 2, 2]) == "\u2584" * 3 and sparkline([]) == ""
    assert len(sparkline(list(range(100)), width=24)) == 24
    assert text_lines(Chart("gn", [("a", [1, 2], [1.0, float("nan")])])) == ["gn a: \u2584 1..1 (last 1)"]
    assert charts_mod.render_png(CHART) is None  # matplotlib "missing" (fixture)


def test_render_png_with_matplotlib(monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.undo()
    png = charts_mod.render_png(Chart("loss", [("a", [1e9, 1e9 + 60, 1e9 + 120], [0.9, 0.8, 0.7])], time_axis=True,
                                      hlines=[("ref", 0.75)]))
    assert png.startswith(b"\x89PNG")
    assert charts_mod.render_png(Chart("empty", [("a", [1], [1.0])])) is None


def test_multipart_body():
    body, ctype = multipart({"image_type": "message"}, {"image": ("c.png", b"PNGDATA", "image/png")})
    boundary = ctype.split("boundary=")[1]
    assert ctype.startswith("multipart/form-data; boundary=yeto-")
    assert b'name="image_type"\r\n\r\nmessage\r\n' in body
    assert b'filename="c.png"\r\nContent-Type: image/png\r\n\r\nPNGDATA\r\n' in body
    assert body.endswith(f"--{boundary}--\r\n".encode())


def test_lark_ws_backoff_resets_after_a_long_healthy_connection(monkeypatch):
    """A connection that stayed up for an hour must reconnect after the base delay,
    not after the backoff accumulated by earlier fast failures."""
    import types

    now = [0.0]
    calls = [0]
    stop = threading.Event()

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            calls[0] += 1
            if calls[0] == 5:
                now[0] += 3600.0  # healthy for an hour, then the socket drops
                return
            raise RuntimeError("connect refused")

    sleeps: list[float] = []

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) >= 6:
            stop.set()

    lark = types.ModuleType("lark_oapi")
    ws = types.ModuleType("lark_oapi.ws")
    wsc = types.ModuleType("lark_oapi.ws.client")
    ws.Client = FakeClient
    ws.client = wsc
    lark.ws = ws
    lark.LogLevel = types.SimpleNamespace(WARNING=0)
    for name, mod in (("lark_oapi", lark), ("lark_oapi.ws", ws), ("lark_oapi.ws.client", wsc)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(lark_ws, "build_handler", lambda ch: None)
    ch = types.SimpleNamespace(app_id="a", app_secret="s", domain="d")
    t = lark_ws.start(ch, stop=stop, sleep=fake_sleep, clock=lambda: now[0])
    t.join(timeout=10)
    assert not t.is_alive()
    assert sleeps == [5.0, 10.0, 20.0, 40.0, 5.0, 10.0]
