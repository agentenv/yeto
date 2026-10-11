from yeto.rl.harness.codex.compaction_bridge import qwen3_cyber_compaction_config

def test_qwen3_cyber_profile_reserves_context_and_segments():
    c=qwen3_cyber_compaction_config()
    assert c.context_budget==24576
    assert c.t_comp==4096
    assert c.max_compactions==3
