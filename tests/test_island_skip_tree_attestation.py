from yeto.rl.adapters.miles import island_entry


def test_skip_only_on_exact_one():
    assert island_entry.skip_tree_attestation({island_entry.SKIP_ATTESTATION_ENV: "1"})
    for value in ("", "0", "true", "yes", " 1"):
        assert not island_entry.skip_tree_attestation({island_entry.SKIP_ATTESTATION_ENV: value})
    assert not island_entry.skip_tree_attestation({})


def test_default_reads_process_env(monkeypatch):
    monkeypatch.delenv(island_entry.SKIP_ATTESTATION_ENV, raising=False)
    assert not island_entry.skip_tree_attestation()
    monkeypatch.setenv(island_entry.SKIP_ATTESTATION_ENV, "1")
    assert island_entry.skip_tree_attestation()
