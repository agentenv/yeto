"""The resource ledger is append only: the same id may be recorded ``create`` more than
once (a pool adopted by a restarted controller deploys its Modal app again, same app id).
Readers decide on the folded state (``ResourceLedger.resources``)."""

from __future__ import annotations


def test_a_pool_adopted_after_a_restart_records_its_app_twice_and_folds_to_one_released(tmp_path, monkeypatch):
    """INT2 bug 21: a restarted controller adopting a Modal pool deploys its app again (the
    same app id): the append-only ledger holds two ``create`` lines for one app. Readers
    decide on the folded state: one app, its first creation time, released, not open."""
    from types import SimpleNamespace  # noqa: PLC0415

    from yeto.cost.ledger import LEDGER_ENV, ResourceLedger  # noqa: PLC0415
    from yeto.launch.pool.modal_provider import ModalSdkClient  # noqa: PLC0415

    path = tmp_path / "resources.jsonl"
    monkeypatch.setenv(LEDGER_ENV, str(path))
    t = [1000.0]
    ledger = ResourceLedger(path, clock=lambda: t[0])
    monkeypatch.setattr("yeto.cost.ledger.current", lambda environ=None: ledger)

    class App:
        app_id = None

        def deploy(self, name):
            self.app_id = "ap-same"  # Modal: a deploy of the same app name keeps its id

    fn = SimpleNamespace(spawn=lambda payload: SimpleNamespace(object_id=f"fc-{payload['shard']}"))
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=lambda app, name: fn))
    for shard in ("first-run-up", "adopting-run-up"):  # one client per run up process
        client = ModalSdkClient({"image": "i", "secret": "s"}, run_id="run-m", modal=modal)
        client.build_app = App
        client.spawn({"shard": shard})
        t[0] += 600
    ledger.released("modal_app", "ap-same", "yeto-rollout-run-m")  # the pool's stop / cleanup
    raw = [r["op"] for r in ledger.records() if r["kind"] == "modal_app" and r.get("id") == "ap-same"]
    assert raw == ["create", "create", "release"]  # append only: the raw lines are not deduped
    apps = [r for r in ledger.resources() if r.kind == "modal_app"]
    assert len(apps) == 1 and apps[0].id == "ap-same"
    assert apps[0].created_at == 1000.0 and apps[0].released_at == 2200.0 and not apps[0].open
    assert not [r for r in ledger.open_resources() if r.kind == "modal_app"]
