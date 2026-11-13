"""``--modal-env`` parsing never quotes a malformed item (often a pasted secret)."""

from __future__ import annotations

import pytest

from yeto.launch.modal_islands import parse_modal_env


def test_modal_env_items():
    assert parse_modal_env(["A=1", "B=x=y"]) == {"A": "1", "B": "x=y"}


@pytest.mark.parametrize("item", ["hf_0123456789abcdefSECRET", "=hf_0123456789abcdefSECRET"])
def test_malformed_modal_env_item_is_not_quoted(item):
    with pytest.raises(ValueError) as err:
        parse_modal_env(["A=1", item])
    assert "hf_0123456789abcdefSECRET" not in str(err.value) and "item 2" in str(err.value)
