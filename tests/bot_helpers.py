"""Fake dashboard views (``/api/view`` shape, ``sft`` section) and fake HTTP for the bot tests."""

import copy

from yeto.bot.channels import Channel

T0 = 1_790_000_000.0
REFS = {"sft_ref_step_s": 571.0, "sft_hbm_limit_gib": 53.9, "sft_syncer_rss_limit_bytes": 1.4e12,
        "sft_slow_factor": 1.1}
# column names the view sends with the positional rows (sft.cols, sft.syncer.merge_cols)
STEP_COLS = ("step", "ts", "loss", "grad_norm", "lr", "step_s", "tps", "tokens", "target_tokens",
             "sync_s", "export_s", "apply_s", "hbm_gib", "mfu", "pushes", "applies")
MERGE_COLS = ("ts", "step", "fragment", "merge_s", "gnorm", "round_s", "quorum_s", "grace_s", "wait_s")


def row(step, *, loss=0.7, gn=1.0, lr=1e-5, step_s=570.0, hbm=52.0, ts=None):
    # in STEP_COLS order
    return [step, T0 + 570 * step if ts is None else ts, loss, gn, lr, step_s, 1.75e4, 1e7, 4e6, 0.5, 0.0, 0.0,
            hbm, 14.5, 0, 0]


def island(iid, steps=10, status="ok", total=100, **row_kw):
    rows = [row(s, **row_kw) for s in range(1, steps + 1)]
    return {"id": str(iid), "name": f"isl{iid}", "nodes": "a,b", "layout": "TP4/EP8/PP3", "cards": 24,
            "dataset": "v2", "gbs": 384, "steps_total": total, "status": status, "last_ts": rows[-1][1],
            "age_s": 30.0, "last_push": {"step": 9, "ts": T0 + 5130, "export_s": 3.0, "sync_s": 0.5},
            "last_apply": None, "rows": rows, "host": {}}


def view(n_islands=2, *, now=None, outer_steps=8, round_s=600.0, events=(), rss=9e11, eval_rows=(), alerts=(),
         rounds=()):
    outer = [[T0 + round_s * k, k, k % 2, 2.0, False, 2] for k in range(1, outer_steps + 1)]
    merges = [[T0 + round_s * k, k, k % 2, 4.0, 1.3, round_s, 30.0, 10.0, round_s - 4.0]
              for k in range(1, outer_steps + 1)]
    last = outer[-1][0] if outer else T0
    return {
        "overview": {"run": "sft-test", "alerts": list(alerts), "islands": [], "global_status": {"text": "ok"}},
        "sft": {
            "now": last + 60 if now is None else now, "refs": dict(REFS), "cols": list(STEP_COLS),
            "islands": [island(i) for i in range(n_islands)],
            "syncer": {"meta": {}, "total_steps": 960, "outer_step": outer_steps, "eta_ts": None,
                       "outer": outer, "merges": merges, "merge_cols": list(MERGE_COLS), "rss": [[last, rss]],
                       "events": [list(e) for e in events], "log_seen": True},
            "eval": {"rows": [dict(r) for r in eval_rows]},
        },
        "rounds_syncer": [dict(r) for r in rounds],
    }


def round_row(step, missed=(), resend=0):
    # subset of yeto.dashboard.reducer.Reducer.rounds() rows
    return {"round": step, "missed": [str(m) for m in missed], "resend": resend, "bad": bool(missed or resend)}


def eval_row(step, loss=0.70, d_base=-0.05, d_ck=-0.01, ref="ckpt100"):
    # every vs_<name> key is a reference comparison; the bot labels it <name>
    return {"step": step, "loss": loss, "ppl": 2.01,
            "vs_base": {"dloss": d_base, "mean": d_base, "se": 0.0015, "better": 0.9, "ref_loss": loss - d_base},
            f"vs_{ref}": {"dloss": d_ck, "mean": d_ck, "se": 0.001, "better": 0.6, "ref_loss": loss - d_ck},
            "vs_prev": None}


def clone(v):
    return copy.deepcopy(v)


def relayout(v, cols=None, merge_cols=None):
    """Copy of ``v`` whose island / merge rows use another column layout: names
    reordered, dropped, or new ones added (filled with None)."""
    v = clone(v)
    sft = v["sft"]

    def remap(rows, old, new):
        return [[r[old.index(n)] if n in old and old.index(n) < len(r) else None for n in new] for r in rows]

    if cols is not None:
        for isl in sft["islands"]:
            isl["rows"] = remap(isl["rows"], sft["cols"], cols)
        sft["cols"] = list(cols)
    if merge_cols is not None:
        syn = sft["syncer"]
        syn["merges"] = remap(syn["merges"], syn["merge_cols"], merge_cols)
        syn["merge_cols"] = list(merge_cols)
    return v


class FakeClock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


class FakeHttp:
    """Records requests; answers from a list of (predicate substring, response or exception)."""

    def __init__(self, routes=None):
        self.calls = []
        self.routes = list(routes or [])

    def __call__(self, method, url, *, body=None, headers=None, timeout=0, files=None):
        self.calls.append({"method": method, "url": url, "body": body, "headers": headers or {}, "files": files})
        for needle, resp in self.routes:
            if needle in url:
                if isinstance(resp, list):  # a queue of responses for this route
                    item = resp.pop(0) if len(resp) > 1 else resp[0]
                else:
                    item = resp
                if isinstance(item, BaseException):
                    raise item
                return copy.deepcopy(item(body) if callable(item) else item)
        return {"ok": True, "code": 0, "result": []}

    def bodies(self, needle):
        return [c["body"] for c in self.calls if needle in c["url"]]


class FakeChannel(Channel):
    """Channel double that records sent messages and button acknowledgements."""

    def __init__(self, name="fake", chat="", can_receive=True):
        super().__init__(chat)
        self.name = name
        self.can_receive = can_receive
        self.sent = []
        self.answers = []

    def send(self, msg, chat=None):
        target = chat or self.report_chat
        if not target:
            return False
        self.sent.append((target, msg))
        return True

    def answer(self, cmd, text):
        self.answers.append(text)
