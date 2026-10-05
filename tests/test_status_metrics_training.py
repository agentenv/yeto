"""fleet-dashboard 1.5: `yeto status --tape` shows training fields; missing -> 无数据."""

from pathlib import Path

from yeto.status_metrics import render_tape_summary

FX = Path(__file__).parent / "fixtures" / "dashboard"


def test_old_syncer_tape_summary_is_unchanged(tmp_path):
    lines = render_tape_summary(FX / "kill44" / "syncer.jsonl")
    assert not [line for line in lines if line.startswith("TRAIN")]
    assert any(line.startswith("ROUNDS 4") for line in lines)


def test_ports_learner_tape_shows_missing_fields_as_no_data():
    lines = render_tape_summary(FX / "s9-m4x1" / "rl-island-0.jsonl")
    train = [line for line in lines if line.startswith("TRAIN")]
    assert train == [
        "TRAIN island=0 round=2 train_step=无数据 reward=0.4688 loss=无数据 pg_loss=无数据 "
        "grad_norm=0.5178 kl=无数据 entropy=无数据 clipfrac=无数据 lr=8.75e-06 adv_mean=无数据 "
        "resp_len=无数据 trunc=无数据 tok/s=无数据"
    ]


def test_new_fields_are_rendered(tmp_path):
    tape = tmp_path / "t.jsonl"
    tape.write_text(
        '{"event":"rl_round_trained","island_id":1,"rollout_id":0,"train_step":7,'
        '"train_metrics":{"loss":0.25,"entropy":1.5,"ppo_kl":0.003},"adv_mean":0.0,'
        '"resp_len_mean":612.0,"truncated_frac":0.02,"tok_per_s":5100.0}\n')
    (line,) = [x for x in render_tape_summary(tape) if x.startswith("TRAIN")]
    assert "train_step=7" in line and "loss=0.25" in line and "entropy=1.5" in line
    assert "kl=0.003" in line and "adv_mean=0" in line and "tok/s=5100" in line
