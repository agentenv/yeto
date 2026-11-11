"""Bot configuration, read from environment variables only.

Thresholds left unset fall back to the dashboard: the HBM line and the syncer
RSS cap come from ``sft.refs`` in ``/api/view`` (the dashboard's
``--thresholds`` file changes both), the HBM warning sits at a fraction of
that HBM line, and the consecutive-missed-rounds count
reuses ``yeto.dashboard.alerts.DEFAULT_THRESHOLDS``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from yeto.dashboard.alerts import DEFAULT_THRESHOLDS


def _str(env: dict, key: str, default: str = "") -> str:
    return (env.get(key) or default).strip()


def _float(env: dict, key: str, default: float | None) -> float | None:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None


def _bool(env: dict, key: str, default: bool = False) -> bool:
    raw = (env.get(key) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _list(env: dict, key: str) -> list[str]:
    return [x.strip() for x in (env.get(key) or "").replace(";", ",").split(",") if x.strip()]


def _telegram_ids(items: list[str]) -> list[str]:
    bad = [x for x in items if not re.fullmatch(r"\d+", x)]
    if bad:
        raise ValueError(f"TELEGRAM_ALLOWED_USERS takes numeric Telegram user ids (the from.id of a message),"
                         f" got {bad}; usernames and display names are not trusted")
    return items


def parse_hours(spec: str) -> tuple[int, int] | None:
    """``"23-8"`` / ``"23:00-08:00"`` -> (start minute, end minute) of the day; None when empty."""
    spec = (spec or "").strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{1,2})(?::(\d\d))?\s*-\s*(\d{1,2})(?::(\d\d))?", spec)
    if not m:
        raise ValueError(f"quiet hours must look like 23-8 or 23:00-08:00, got {spec!r}")
    a, b = int(m.group(1)) * 60 + int(m.group(2) or 0), int(m.group(3)) * 60 + int(m.group(4) or 0)
    if a >= 24 * 60 or b > 24 * 60:
        raise ValueError(f"quiet hours out of range: {spec!r}")
    return a, b


def parse_clock(spec: str) -> int | None:
    """``"09:00"`` -> minute of the day; None when empty."""
    spec = (spec or "").strip()
    if not spec:
        return None
    m = re.fullmatch(r"(\d{1,2})(?::(\d\d))?", spec)
    if not m or int(m.group(1)) > 23:
        raise ValueError(f"time of day must look like 09:00, got {spec!r}")
    return int(m.group(1)) * 60 + int(m.group(2) or 0)


@dataclass
class Thresholds:
    # None: resolved against the dashboard refs at evaluation time
    hbm_limit_gib: float | None = None  # P0 at or above (refs sft_hbm_limit_gib)
    hbm_warn_gib: float | None = None  # P1 above; None: hbm_warn_frac x the HBM limit (no limit: no warning)
    # 0.965 x 53.9 GiB keeps the warning just above a healthy ~52 GiB peak on the reference card
    hbm_warn_frac: float = 0.965
    rss_warn_bytes: float | None = None  # P1 above (refs sft_syncer_rss_limit_bytes)
    host_avail_min_bytes: float = 100e9  # P1 below (per node, when the dashboard has host samples)
    disk_free_min_frac: float = 0.05  # P0 below (only if the view ever carries disk samples)
    loss_window: int = 20
    loss_min_points: int = 5
    loss_sigma: float = 3.0  # P1: loss > mean + sigma * std of the window
    grad_norm_max: float = 1.0  # P1 above
    missed_rounds: int = int(DEFAULT_THRESHOLDS["missed_consecutive_warn"])  # P1: N rounds in a row
    grace_s: float = 600.0  # syncer grace window
    stuck_round_s: float | None = None  # P0; None: median round interval + 2 x grace
    dashboard_down_after: int = 3
    forward_dashboard_alerts: bool = True  # dashboard sev 0 -> P0


@dataclass
class Policy:
    p0_renotify_s: float = 1800.0
    p1_cooldown_s: float = 3600.0
    ack_mute_s: float = 7200.0
    quiet_hours: tuple[int, int] | None = None  # only P0 inside
    progress_interval: float = 7200.0
    digest_at: int | None = 9 * 60  # minute of the local day; None disables


@dataclass
class Config:
    dashboard_url: str = "http://127.0.0.1:8787"
    dashboard_link: str = ""  # URL for the "Open dashboard" button (e.g. through a tunnel)
    dashboard_timeout: float = 10.0
    poll_interval: float = 60.0
    dry_run: bool = False
    title_prefix: str = ""  # e.g. "[staging] " to tell several bots / runs apart
    state_file: str = ""
    audit_log: str = "yeto-bot-audit.jsonl"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_api_base: str = "https://api.telegram.org"
    telegram_owners: list[str] = field(default_factory=list)  # usernames to @ on P0
    # numeric Telegram user ids (from.id); usernames are not accepted since they can be
    # released and re-registered by someone else. Empty: anyone in the report chat.
    telegram_allowed: list[str] = field(default_factory=list)
    lark_app_id: str = ""
    lark_app_secret: str = ""
    lark_chat_id: str = ""
    lark_domain: str = "https://open.larksuite.com"  # Feishu (China): https://open.feishu.cn
    lark_mode: str = "auto"  # auto | ws | app | webhook
    lark_card_buttons: bool = True  # callback buttons need card.action.trigger on the long connection
    lark_owners: list[str] = field(default_factory=list)  # open_ids to @ on P0
    lark_allowed: list[str] = field(default_factory=list)  # open_ids; empty: anyone in the chat
    lark_webhook_url: str = ""
    lark_webhook_secret: str = ""
    lark_token_refresh_margin: float = 600.0
    th: Thresholds = field(default_factory=Thresholds)
    policy: Policy = field(default_factory=Policy)

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Config":
        e = dict(os.environ if env is None else env)
        mode = _str(e, "LARK_MODE", "auto").lower()
        if mode not in ("auto", "ws", "app", "webhook"):
            raise ValueError(f"LARK_MODE must be auto, ws, app or webhook, got {mode!r}")
        d = Thresholds()
        th = Thresholds(
            hbm_limit_gib=_float(e, "YETO_BOT_HBM_LIMIT_GIB", None),
            hbm_warn_gib=_float(e, "YETO_BOT_HBM_WARN_GIB", None),
            hbm_warn_frac=_float(e, "YETO_BOT_HBM_WARN_FRAC", d.hbm_warn_frac),
            rss_warn_bytes=_float(e, "YETO_BOT_RSS_WARN_BYTES", None),
            host_avail_min_bytes=_float(e, "YETO_BOT_HOST_AVAIL_MIN_BYTES", d.host_avail_min_bytes),
            disk_free_min_frac=_float(e, "YETO_BOT_DISK_FREE_MIN_FRAC", d.disk_free_min_frac),
            loss_window=int(_float(e, "YETO_BOT_LOSS_WINDOW", d.loss_window)),
            loss_min_points=int(_float(e, "YETO_BOT_LOSS_MIN_POINTS", d.loss_min_points)),
            loss_sigma=_float(e, "YETO_BOT_LOSS_SIGMA", d.loss_sigma),
            grad_norm_max=_float(e, "YETO_BOT_GRAD_NORM_MAX", d.grad_norm_max),
            missed_rounds=int(_float(e, "YETO_BOT_MISSED_ROUNDS", d.missed_rounds)),
            grace_s=_float(e, "YETO_BOT_GRACE_S", d.grace_s),
            stuck_round_s=_float(e, "YETO_BOT_STUCK_ROUND_S", None),
            dashboard_down_after=int(_float(e, "YETO_BOT_DASHBOARD_DOWN_AFTER", d.dashboard_down_after)),
            forward_dashboard_alerts=_bool(e, "YETO_BOT_FORWARD_DASHBOARD_ALERTS", True),
        )
        p = Policy()
        digest = _str(e, "YETO_BOT_DIGEST_AT", "09:00")
        policy = Policy(
            p0_renotify_s=_float(e, "YETO_BOT_P0_RENOTIFY_S", p.p0_renotify_s),
            p1_cooldown_s=_float(e, "YETO_BOT_P1_COOLDOWN_S", p.p1_cooldown_s),
            ack_mute_s=_float(e, "YETO_BOT_ACK_MUTE_S", p.ack_mute_s),
            quiet_hours=parse_hours(_str(e, "YETO_BOT_QUIET_HOURS")),
            progress_interval=_float(e, "YETO_BOT_PROGRESS_INTERVAL", p.progress_interval),
            digest_at=None if digest.lower() in ("off", "none") else parse_clock(digest),
        )
        return cls(
            dashboard_url=_str(e, "YETO_DASHBOARD_URL", "http://127.0.0.1:8787").rstrip("/"),
            dashboard_link=_str(e, "YETO_DASHBOARD_LINK"),
            dashboard_timeout=_float(e, "YETO_DASHBOARD_TIMEOUT", 10.0),
            poll_interval=_float(e, "YETO_BOT_POLL_INTERVAL", 60.0),
            dry_run=_bool(e, "YETO_BOT_DRY_RUN"),
            title_prefix=e.get("YETO_BOT_TITLE_PREFIX") or "",
            state_file=_str(e, "YETO_BOT_STATE_FILE"),
            audit_log=_str(e, "YETO_BOT_AUDIT_LOG", "yeto-bot-audit.jsonl"),
            telegram_token=_str(e, "TELEGRAM_BOT_TOKEN"),
            telegram_chat_id=_str(e, "TELEGRAM_CHAT_ID"),
            telegram_api_base=_str(e, "TELEGRAM_API_BASE", "https://api.telegram.org").rstrip("/"),
            telegram_owners=[x.lstrip("@") for x in _list(e, "TELEGRAM_OWNERS")],
            telegram_allowed=_telegram_ids(_list(e, "TELEGRAM_ALLOWED_USERS")),
            lark_app_id=_str(e, "LARK_APP_ID"),
            lark_app_secret=_str(e, "LARK_APP_SECRET"),
            lark_chat_id=_str(e, "LARK_CHAT_ID"),
            lark_domain=_str(e, "LARK_DOMAIN", "https://open.larksuite.com").rstrip("/"),
            lark_mode=mode,
            lark_card_buttons=_bool(e, "LARK_CARD_BUTTONS", True),
            lark_owners=_list(e, "LARK_OWNER_OPEN_IDS") or _list(e, "LARK_OWNER_OPEN_ID"),
            lark_allowed=_list(e, "LARK_ALLOWED_OPEN_IDS"),
            lark_webhook_url=_str(e, "LARK_WEBHOOK_URL"),
            lark_webhook_secret=_str(e, "LARK_WEBHOOK_SECRET"),
            lark_token_refresh_margin=_float(e, "LARK_TOKEN_REFRESH_MARGIN", 600.0),
            th=th,
            policy=policy,
        )
