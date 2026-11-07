"""yeto.bot: message formatting, command routing, chat / user authorization, schedules,
decision audit log, config."""

import json
import time

import pytest

from yeto.bot.app import Bot, parse_command, parse_duration
from yeto.bot.channels import Command
from yeto.bot.config import Config, Policy
from yeto.bot.render import (ckpt_message, eval_message, find_island, island_message, merge_health, merge_message,
                             problem_counts, res_message, status_message, strip_bold)

from bot_helpers import (MERGE_COLS, STEP_COLS, T0, FakeChannel, FakeClock, clone, eval_row, relayout, round_row,
                         row, view)


# -- formatting ---------------------------------------------------------------------

def test_progress_card():
    v = view(events=[[T0, "rejected", 1, "log", None]], eval_rows=[eval_row(64), eval_row(96, loss=0.69)],
             rounds=[round_row(1), round_row(2, missed=[1]), round_row(3)])
    v["sft"]["syncer"]["eta_ts"] = T0 + 86400
    v["sft"]["islands"][1]["status"] = "warn"
    msg = status_message(v, alerts_line="Alerts active P0 0 | P1 1")
    text = msg.plain()
    assert msg.title == "yeto progress: sft-test"
    assert text.splitlines()[1].startswith("Outer 8/960 (0.8%) | ETA ")
    assert "isl0 10/100 | loss 0.7000 | 9.5min/step" in text
    assert "isl1 10/100 | loss 0.7000 | 9.5min/step | warn" in text
    assert ("Merge direct-in-round 2/3 (67%) | median round 10min (quorum 30s, grace 10s, merge 4s)"
            " | missed 1 x1") in text
    assert "Eval step 96 loss 0.6900 | vs base -0.0500 | vs ckpt100 -0.0100" in text
    assert text.endswith("Alerts active P0 0 | P1 1")
    assert [c.title for c in msg.charts] == ["loss, last 6 h"]
    assert ("Merge", "/merge") in msg.buttons


def test_progress_uses_latest_logged_values_and_rl_fallback():
    v = view(1)
    v["sft"]["islands"][0]["rows"].append(row(11, loss=None, gn=None, lr=None, hbm=None))
    assert "isl0 11/100 | loss 0.7000" in status_message(v).plain()
    rl = {"overview": {"run": "rl", "global_status": {"text": "all healthy"},
                       "islands": [{"id": "0", "status": "ok", "round": 3, "last_event_age_s": 5}], "alerts": []}}
    assert "island 0 ok | round 3" in status_message(rl).plain()


def test_island_lookup_detail_and_charts():
    v = view(3)
    sft = v["sft"]
    assert find_island(sft, "2")["id"] == "2" and find_island(sft, "ISL1")["id"] == "1"
    assert find_island(sft, "island 0")["id"] == "0"
    msg = island_message(v, "isl1", hbm_limit=53.9)
    text = msg.plain()
    assert msg.title == "island 1: isl1" and "layout TP4/EP8/PP3" in text and "HBM 52/53.9 GiB" in text
    assert [c.title for c in msg.charts] == ["isl1 loss", "isl1 grad_norm", "isl1 step time s"]
    assert island_message(v, "nope").level == "warn"


def test_merge_res_ckpt_eval_messages(tmp_path):
    v = view(events=[[T0 + 1, "rejected", 1, "log", None], [T0 + 2, "checkpoint", None, "log", 12.5]],
             rounds=[round_row(1), round_row(2, missed=[0], resend=1)], eval_rows=[eval_row(64, loss=0.72),
                                                                                    eval_row(96)])
    syn = v["sft"]["syncer"]
    syn["outer"][3][4] = True
    syn["merges"][5][5] = 3000.0
    m = merge_message(v).plain()
    assert "Timed-out rounds (1)" in m and "Slow rounds (> 2x median 10min): 1" in m
    assert "rejected 1 | timeout 0 | not carried 0" in m and "round 2: missed 0 | resend 1" in m
    assert merge_health(v)["direct"] == 1
    v["sft"]["islands"][0]["host"] = {"node-a": [[T0, 80e9, 1.5e12]]}
    r = res_message(v, hbm_limit=53.9, hbm_warn=52.0, rss_warn=1.4e12, host_min=100e9).plain()
    assert "isl0: last 52 | max of 20 steps 52 GiB" in r and "isl0 node-a: 80.00 GB of 1.50 TB" in r
    assert "Syncer RSS 900.00 GB (warn > 1.40 TB)" in r
    bot, ch, client = make_bot(tmp_path, chat="1", v=clone(v))
    assert bot.handle(ch, cmd("/res")).plain().splitlines()[1] == "HBM peak (warn > 52, limit 53.9 GiB)"
    del client.v["sft"]["refs"]["sft_hbm_limit_gib"]
    assert bot.handle(ch, cmd("/res")).plain().splitlines()[1] == "HBM peak (warn > -, limit - GiB)"
    c = ckpt_message(v).plain()
    assert "Syncer checkpoint writes (1)" in c and "written in 12s" in c
    e = eval_message(v)
    assert e.title == "yeto eval: step 96"
    assert "vs base: dloss -0.0500 (ref 0.7500) | paired se 0.0015 | better on 90%" in e.plain()
    assert "64 | 0.7200 | -0.0500 | -0.0100 | -" in e.plain()
    assert [c.title for c in e.charts] == ["held-out eval loss"]
    assert problem_counts(v["sft"])["rejected"] == 1


def test_messages_resolve_columns_by_name(tmp_path):
    v = view(2, eval_rows=[eval_row(96)], rounds=[round_row(1)])
    v["sft"]["islands"][0]["host"] = {"node-a": [[T0, 80e9, 1.5e12]]}
    v["sft"]["syncer"]["merges"][5][5] = 3000.0  # one slow round
    shuffled = relayout(v, cols=list(reversed(STEP_COLS)), merge_cols=list(reversed(MERGE_COLS)))
    bot, ch, client = make_bot(tmp_path, chat="1", v=v)

    def texts():
        out = {t: bot.handle(ch, cmd(t)).plain() for t in ("/status", "/island 0", "/merge", "/res")}
        out["digest"] = bot.digest(client.v).plain()
        return out

    want = texts()
    assert "HBM 52/53.9 GiB | tokens/s 1.75e+04 | MFU 14.5% | lr 1e-05" in want["/island 0"]
    assert "Slow rounds (> 2x median 10min): 1" in want["/merge"]
    client.v = shuffled
    assert texts() == want
    client.v = relayout(v, cols=[c for c in STEP_COLS if c not in ("hbm_gib", "mfu")], merge_cols=["ts", "step"])
    island = bot.handle(ch, cmd("/island 0")).plain()
    assert "HBM -/53.9 GiB" in island and "MFU -%" in island and "| loss 0.7000 |" in island
    assert "isl0: last - | max of 20 steps - GiB" in bot.handle(ch, cmd("/res")).plain()
    assert "Slow rounds" not in bot.handle(ch, cmd("/merge")).plain()


def test_eval_references_are_every_vs_key():
    rows = [eval_row(64, ref="warmstart", d_ck=-0.02), eval_row(96, loss=0.69, ref="warmstart")]
    rows[1]["vs_prev"] = {"dloss": -0.01, "se": 0.001, "better": 0.55, "ref_loss": 0.70}
    rows[1]["vs_sft_v1"] = {"dloss": 0.03, "ref_loss": 0.66}  # a reference that only newer points carry
    v = view(eval_rows=rows)
    assert ("Eval step 96 loss 0.6900 | vs base -0.0500 | vs warmstart -0.0100 | vs sft_v1 +0.0300"
            " | vs prev -0.0100") in status_message(v).plain()
    e = eval_message(v)
    text = e.plain()
    assert "step | loss | vs base | vs warmstart | vs sft_v1 | vs prev" in text
    assert "64 | 0.7000 | -0.0500 | -0.0200 | - | -" in text and "vs sft_v1: dloss +0.0300 (ref 0.6600)" in text
    assert "ckpt100" not in text
    assert [s[0] for s in e.charts[0].series] == ["eval loss", "base ref", "warmstart ref", "sft_v1 ref"]


# -- commands and authorization -----------------------------------------------------

class FakeClient:
    def __init__(self, v):
        self.v = v
        self.fail = None

    def view(self):
        if self.fail:
            raise self.fail
        return self.v

    def bad_rounds(self):
        return []


def make_bot(tmp_path, chat="", v=None, allowed=None, nl=None, clock=None, **cfg_kw):
    cfg_kw.setdefault("policy", Policy(digest_at=None))
    cfg = Config(state_file=str(tmp_path / "state.json"), audit_log=str(tmp_path / "audit.jsonl"), **cfg_kw)
    ch = FakeChannel("telegram", chat)
    ch.allowed = list(allowed or [])
    client = FakeClient(v or view(eval_rows=[eval_row(96)]))
    return Bot(cfg, client, [ch], clock=clock or FakeClock(), nl=nl), ch, client


def cmd(text, chat="1", user="al", button=False, user_id="1001"):
    return Command("telegram", chat, text, user, button, None, user_id)


def test_parse_command_and_duration():
    assert parse_command("/Island@yeto_bot  2 ") == ("/island", "2")
    assert parse_duration("2h") == 7200 and parse_duration("30m") == 1800 and parse_duration("45") == 2700
    assert parse_duration("soon") is None


def test_start_registers_then_only_report_chat(tmp_path):
    bot, ch, _ = make_bot(tmp_path, allowed=["7"])
    assert bot.handle(ch, cmd("/status", "111")).title == "not registered"
    assert bot.handle(ch, cmd("hello", "111")) is None
    assert bot.handle(ch, cmd("/start", "111", user_id="8")) is None and ch.report_chat == ""  # not allowlisted
    assert "Registered" in bot.handle(ch, cmd("/start", "111", user_id="7")).plain() and ch.report_chat == "111"
    assert bot.handle(ch, cmd("/start", "222", user_id="7")) is None  # a second chat cannot take over
    assert bot.handle(ch, cmd("/status", "222", button=True, user_id="7")) is None
    assert ch.answers[-1] == "Not allowed"
    assert all(target == "111" for target, _ in ch.sent)
    assert bot.handle(ch, cmd("/status", "111", user_id="7")).title == "yeto progress: sft-test"
    bot2, ch2, _ = make_bot(tmp_path)  # registration survives a restart via the state file
    assert ch2.report_chat == "111"


def test_start_without_allowlist_never_registers(tmp_path):
    bot, ch, _ = make_bot(tmp_path)
    ch.chat_env, ch.allow_env = "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_USERS"
    reply = bot.handle(ch, cmd("/start", "111", user_id="7"))
    assert reply.title == "registration disabled" and ch.report_chat == ""
    assert "Set TELEGRAM_CHAT_ID or TELEGRAM_ALLOWED_USERS" in reply.plain()
    assert not (tmp_path / "state.json").exists()


def test_user_allowlist(tmp_path):
    bot, ch, _ = make_bot(tmp_path, chat="1", allowed=["ou_owner", "42"])
    assert bot.handle(ch, cmd("/status", user="ou_other", user_id="ou_other")) is None
    assert bot.handle(ch, cmd("/status", user="ou_owner", user_id="ou_owner")).title.startswith("yeto progress")
    assert bot.handle(ch, cmd("/status", user="tg_name", user_id="42")) is not None
    assert bot.handle(ch, cmd("/status", user="42", user_id="7")) is None  # display names never authorize


def test_cli_command_is_trusted_past_the_allowlist(tmp_path, monkeypatch):
    from yeto.bot import __main__ as main_mod

    ch = FakeChannel("telegram", "1")
    ch.allowed = ["42"]
    monkeypatch.setattr(main_mod.Config, "from_env", classmethod(
        lambda cls: Config(audit_log=str(tmp_path / "audit.jsonl"), policy=Policy(digest_at=None))))
    monkeypatch.setattr(main_mod, "build_channels", lambda cfg, receive: [ch])
    monkeypatch.setattr(main_mod, "DashboardClient", lambda *a, **kw: FakeClient(view()))
    assert main_mod.main(["--command", "/status"]) == 0
    assert [(target, m.title) for target, m in ch.sent] == [("1", "yeto progress: sft-test")]
    bot, ch2, _ = make_bot(tmp_path, chat="1", allowed=["42"])
    assert bot.handle(ch2, Command("telegram", "1", "/status", "cli")) is None  # the flag, not the name


def test_commands_dispatch_and_errors(tmp_path):
    bot, ch, client = make_bot(tmp_path, chat="1")
    for text, title in (("/island 1", "island 1: isl1"), ("/eval", "yeto eval: step 96"), ("/merge", "yeto merge"),
                        ("/res", "yeto resources"), ("/ckpt", "yeto checkpoints"), ("/alerts", "yeto alerts"),
                        ("/help", "yeto training bot")):
        assert bot.handle(ch, cmd(text)).title == title, text
    assert "unknown command /foo" in bot.handle(ch, cmd("/foo")).plain()
    client.fail = OSError("dashboard down")
    reply = bot.handle(ch, cmd("/status"))
    assert reply.title == "command failed" and "dashboard down" in reply.plain()


def test_ack_details_mute_flow(tmp_path):
    clock = FakeClock()
    bot, ch, client = make_bot(tmp_path, chat="1", clock=clock, allowed=["1001"],
                               policy=Policy(digest_at=None, progress_interval=0), dashboard_link="https://d/")
    bot.tick()
    st = clone(client.v)
    st["sft"]["islands"][0]["status"] = "stale"
    client.v = st
    clock.t += 60
    [card] = bot.tick()
    assert card.level == "p0" and ("Ack 2h", "/ack c1") in card.buttons
    det = bot.handle(ch, cmd("/details c1", button=True))
    assert det.title == "details c1" and "isl0 (status stale" in det.plain()
    assert det.plain().count("\n10 | ") == 1  # last 10 steps table
    ack = bot.handle(ch, cmd("/ack c1", user="bob", button=True))
    assert ack.title == "acked 1 alert(s) for 2h" and "isl0 stale" in ack.plain()
    clock.t += 3600
    assert bot.tick() == []  # re-notify held back by the ack
    assert bot.handle(ch, cmd("/ack c99")).title == "unknown card"
    assert bot.handle(ch, cmd("/mute 2h")).title == "muted"
    assert "muted for 2h" in bot.alerts_line()
    assert bot.handle(ch, cmd("/mute later")).title == "usage"
    assert bot.handle(ch, cmd("/unmute")).title == "unmuted"
    assert bot.monitor.mute_until == 0


def test_alert_control_needs_an_allowlist(tmp_path):
    bot, ch, _ = make_bot(tmp_path, chat="1")
    ch.allow_env = "TELEGRAM_ALLOWED_USERS"
    for text in ("/mute 1d all", "/unmute", "/ack all", "/ack c1"):
        reply = bot.handle(ch, cmd(text, button=True))
        assert reply.title == "not allowed" and "TELEGRAM_ALLOWED_USERS" in reply.plain(), text
        assert ch.answers[-1] == "Needs a user allowlist"
    assert bot.monitor.mute_until == 0
    req = bot.request_decision("restart island 2?", [], ["yes", "no"])
    reply = bot.handle(ch, cmd(f"/decide {req.buttons[0][1].split()[1]} yes", button=True))
    assert reply.title == "not allowed" and "records a decision" in reply.plain()
    assert bot.handle(ch, cmd("/alerts")).title == "yeto alerts"  # read-only commands still work
    assert bot.handle(ch, cmd("/status")).title == "yeto progress: sft-test"
    trusted = Command("telegram", "1", "/mute 1h", "cli", trusted=True)
    assert bot.handle(ch, trusted).title == "muted"  # the local CLI is not a chat user
    assert bot.allowlist_warnings() == [
        "telegram: no user allowlist (TELEGRAM_ALLOWED_USERS unset): anyone in the report chat can run read-only"
        " commands; /ack, /mute, /unmute, /decide are refused"]
    ch.allowed = ["1001"]
    assert bot.allowlist_warnings() == []
    assert bot.handle(ch, cmd("/unmute")).title == "unmuted"


def test_progress_schedule_and_daily_digest(tmp_path):
    start = time.mktime((2026, 11, 7, 8, 0, 0, 0, 0, -1))
    clock = FakeClock(start)
    bot, ch, client = make_bot(tmp_path, chat="1", clock=clock,
                               policy=Policy(progress_interval=7200, digest_at=9 * 60))
    assert [m.title for m in bot.tick()] == ["yeto progress: sft-test"]  # startup card
    clock.t += 3600 - 60
    assert bot.tick() == []
    clock.t += 120  # 09:01: digest
    msgs = bot.tick()
    assert [m.title.split(":")[0] for m in msgs] == ["yeto daily digest"]
    assert "Alerts, last 24 h P0 0 | P1 0 | P2 0" in msgs[0].plain()
    clock.t += 3600  # 10:01: the 2 h progress card
    assert [m.title for m in bot.tick()] == ["yeto progress: sft-test"]
    clock.t += 600
    assert bot.tick() == []  # digest once a day
    late = make_bot(tmp_path, chat="1", clock=FakeClock(start + 4 * 3600),
                    policy=Policy(progress_interval=0, digest_at=9 * 60))[0]
    assert late.tick() == []  # started after 09:00: first digest tomorrow


def test_decision_request_and_audit_log(tmp_path):
    bot, ch, _ = make_bot(tmp_path, chat="1", allowed=["1001"])
    msg = bot.request_decision("island 2 stale for 40 min", ["last step 812", "syncer still merging"],
                               ["restart", "wait", "ignore"], "wait")
    req_id = msg.buttons[0][1].split()[1]
    assert msg.level == "decision" and [b[0] for b in msg.buttons] == ["restart", "wait (suggested)", "ignore"]
    assert ch.sent[-1][1] is msg
    reply = bot.handle(ch, cmd(f"/decide {req_id} restart", user="bob", button=True))
    assert reply.title == "decision recorded" and "recorded; operator will act" in reply.plain()
    again = bot.handle(ch, cmd(f"/decide {req_id} wait", user="eve", button=True))
    assert again.title == "decision already recorded" and "restart by bob" in again.plain()
    assert bot.handle(ch, cmd(f"/decide {req_id} reboot")).title == "unknown option"
    assert bot.handle(ch, cmd("/decide dnope wait")).title == "unknown decision"
    recs = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    assert [r["type"] for r in recs] == ["request", "decision"]
    assert recs[1]["option"] == "restart" and recs[1]["user"] == "bob" and recs[1]["chat"] == "1"
    assert recs[1]["user_id"] == "1001"
    with pytest.raises(ValueError):
        bot.request_decision("x", [], ["two words"])
    # a request posted by another process (CLI) is answered by this bot through the shared log
    other, _, _ = make_bot(tmp_path, chat="1")
    req2 = other.decisions.request("lower lr?", [], ["yes", "no"])
    assert bot.handle(ch, cmd(f"/decide {req2.id} no")).title == "decision recorded"


def test_natural_language_hook(tmp_path):
    seen = []

    def nl(text, tools):
        seen.append((text, tools.view()["overview"]["run"], tools.active_alerts()))
        from yeto.bot.render import Message
        return Message("answer", ["fine"])

    bot, ch, _ = make_bot(tmp_path, chat="1", nl=nl)
    assert bot.handle(ch, cmd("how is island 1?")).title == "answer"
    assert seen == [("how is island 1?", "sft-test", [])]
    assert bot.handle(ch, cmd("how is island 1?", chat="9")) is None
    plain, ch2, _ = make_bot(tmp_path, chat="1")
    assert plain.handle(ch2, cmd("hello")) is None  # no handler installed: ignored


def test_no_report_chat_means_no_broadcast(tmp_path):
    bot, ch, _ = make_bot(tmp_path)
    bot.tick()
    assert ch.sent == []


# -- config -------------------------------------------------------------------------

def test_config_from_env_defaults_and_overrides():
    cfg = Config.from_env({})
    assert cfg.dashboard_url == "http://127.0.0.1:8787" and cfg.lark_domain == "https://open.larksuite.com"
    assert cfg.policy.progress_interval == 7200 and cfg.policy.digest_at == 540 and cfg.policy.quiet_hours is None
    assert cfg.policy.p0_renotify_s == 1800 and cfg.policy.p1_cooldown_s == 3600 and cfg.policy.ack_mute_s == 7200
    assert cfg.th.hbm_limit_gib is None and cfg.th.grad_norm_max == 1.0 and cfg.th.grace_s == 600
    assert cfg.th.hbm_warn_gib is None and cfg.th.hbm_warn_frac == 0.965
    assert cfg.lark_card_buttons and cfg.lark_mode == "auto"
    cfg = Config.from_env({"YETO_DASHBOARD_URL": "http://dash:9/", "YETO_BOT_HBM_LIMIT_GIB": "60",
                           "YETO_BOT_QUIET_HOURS": "23-8", "YETO_BOT_DIGEST_AT": "off", "LARK_MODE": "webhook",
                           "LARK_OWNER_OPEN_IDS": "ou_a, ou_b", "TELEGRAM_OWNERS": "@ops_lead",
                           "YETO_BOT_DRY_RUN": "1", "LARK_CARD_BUTTONS": "0"})
    assert cfg.dashboard_url == "http://dash:9" and cfg.th.hbm_limit_gib == 60.0
    assert Config.from_env({"YETO_BOT_HBM_WARN_FRAC": "0.9"}).th.hbm_warn_frac == 0.9
    assert Config.from_env({"YETO_BOT_HBM_WARN_GIB": "70"}).th.hbm_warn_gib == 70.0
    assert cfg.policy.quiet_hours == (1380, 480) and cfg.policy.digest_at is None
    assert cfg.lark_owners == ["ou_a", "ou_b"] and cfg.telegram_owners == ["ops_lead"]
    assert cfg.dry_run and cfg.lark_mode == "webhook" and not cfg.lark_card_buttons
    assert Config.from_env({"TELEGRAM_ALLOWED_USERS": "42, 1001"}).telegram_allowed == ["42", "1001"]
    with pytest.raises(ValueError, match="numeric Telegram user ids"):
        Config.from_env({"TELEGRAM_ALLOWED_USERS": "42,@ops_lead"})
    with pytest.raises(ValueError, match="LARK_MODE"):
        Config.from_env({"LARK_MODE": "smtp"})
    with pytest.raises(ValueError, match="YETO_BOT_POLL_INTERVAL"):
        Config.from_env({"YETO_BOT_POLL_INTERVAL": "soon"})


def test_bold_markers_never_leak_into_plain_text():
    assert "\x02" not in status_message(view()).plain()
    assert strip_bold("\x02a\x03") == "a"


def test_title_prefix_marks_every_outgoing_message(tmp_path):
    bot, ch, _ = make_bot(tmp_path, chat="1", title_prefix="[TEST] ")
    reply = bot.handle(ch, cmd("/eval"))
    assert reply.title == "yeto eval: step 96"  # the returned reply is unprefixed
    assert ch.sent[-1][1].title == "[TEST] yeto eval: step 96"
    bot.tick()
    assert ch.sent[-1][1].title == "[TEST] yeto progress: sft-test"


def test_ack_mutes_event_subjects_too(tmp_path):
    clock = FakeClock()
    bot, ch, client = make_bot(tmp_path, chat="1", clock=clock, allowed=["1001"],
                               policy=Policy(digest_at=None, progress_interval=0, p1_cooldown_s=0))
    ev = [[T0, "rejected", 1, "log", None]]
    client.v = view(events=ev)
    bot.tick()
    client.v = view(events=ev + [[T0 + 10, "rejected", 2, "log", None]])
    clock.t += 60
    [card] = bot.tick()
    assert card.title == "P1 pushes rejected" and ("Ack 2h", "/ack c1") in card.buttons
    assert bot.handle(ch, cmd("/ack c1")).title == "acked 1 alert(s) for 2h"
    client.v = view(events=ev + [[T0 + 10, "rejected", 2, "log", None], [T0 + 20, "rejected", 2, "log", None]])
    clock.t += 60
    assert bot.tick() == []  # the subject is muted for 2 h
