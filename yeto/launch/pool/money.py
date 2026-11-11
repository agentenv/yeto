"""Money of the rollout pools: every provider rate in one unit.

A provider states its rates in ``caps.currency`` (``ProviderCaps``): National
Compute in ``ucr`` (micro-credits), the clouds in ``usd``. The pool's budget
(:class:`~.status.Budget`) keeps one total, in ucr-equivalents (the unit its caps
and its saved state already use); every rate is converted on the way in, so a
run with National Compute and cloud islands in one pool, or in several pools, adds
up correctly. ``max_run_spend_usd`` caps the same total in USD.

The exchange rate is the cost layer's (``yeto.cost.billing``): ``YETO_NC_USD_PER_UCR``
(default 1e-6: one credit, 1e6 ucr, ~ 1 USD).
"""

from __future__ import annotations

import os
from typing import Any

CURRENCIES = ("ucr", "usd")


def usd_per_ucr() -> float:
    from yeto.cost import billing  # noqa: PLC0415

    raw = os.environ.get(billing.NC_USD_PER_UCR_ENV)
    try:
        return float(raw) if raw not in (None, "") else float(billing.DEFAULT_NC_USD_PER_UCR)
    except ValueError:
        return float(billing.DEFAULT_NC_USD_PER_UCR)


def to_ucr(amount: float, currency: str) -> float:
    """``amount`` of ``currency`` in ucr-equivalents."""
    if currency == "ucr":
        return float(amount)
    if currency == "usd":
        return float(amount) / usd_per_ucr()
    raise ValueError(f"currency {currency!r} is not one of {CURRENCIES}")


def to_usd(amount: float, currency: str) -> float:
    if currency == "usd":
        return float(amount)
    if currency == "ucr":
        return float(amount) * usd_per_ucr()
    raise ValueError(f"currency {currency!r} is not one of {CURRENCIES}")


def currency_of(ops: Any, member: Any) -> str:
    """The currency of ``member``'s rate: ``ops.currency(member)``, else the caps of
    its provider (the member's own provider in a ``MultiProvider``), else ``ucr``
    (the pool's historical unit)."""
    own = getattr(ops, "currency", None)
    if callable(own):
        return str(own(member))
    from .provider import caps_of  # noqa: PLC0415

    provider = getattr(ops, "provider", None)
    if provider is None:
        return "ucr"
    split = getattr(provider, "provider_of", None)
    if callable(split):
        try:
            provider = split(member.account)
        except (KeyError, ValueError):
            pass
    caps = getattr(provider, "caps", None)
    return caps_of(provider).currency if caps is not None else "ucr"
