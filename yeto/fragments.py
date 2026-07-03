"""Balanced tensor fragmentation.

Trainable tensors are packed into P fragments by greedy number partitioning:
iterate over tensors in descending size order, placing each into the fragment
with the smallest running total. Embedding-like tensors are isolated into
their own fragment merged with direct averaging (embedding
outer gradients lack the near-orthogonality that motivates RDA); all other
fragments use radial-directional averaging.

The layout must be identical on every learner, so packing is deterministic:
ties break on tensor name and fragment index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MERGE_AVG = 0
MERGE_RDA = 1

_EMBED_MARKERS = ("embed", "wte", "wpe", "lm_head", "shared.weight")


def is_embedding_name(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in _EMBED_MARKERS)


@dataclass
class Fragment:
    merge_mode: int
    tensors: list[tuple[str, int]] = field(default_factory=list)  # (name, numel)

    @property
    def numel(self) -> int:
        return sum(n for _, n in self.tensors)


@dataclass
class FragmentLayout:
    fragments: list[Fragment]

    @property
    def num_fragments(self) -> int:
        return len(self.fragments)

    def tensor_names(self) -> list[str]:
        return [name for f in self.fragments for name, _ in f.tensors]


def build_layout(named_numels: list[tuple[str, int]], num_fragments: int) -> FragmentLayout:
    """Partition (name, numel) pairs into at most `num_fragments` fragments.

    Embedding-like tensors share one avg-merged fragment; the rest are
    bin-packed into the remaining fragments with RDA. If there are fewer
    non-embedding tensors than bins, the fragment count shrinks.
    """
    if num_fragments < 1:
        raise ValueError("num_fragments must be >= 1")
    embed = sorted(
        ((n, s) for n, s in named_numels if is_embedding_name(n)),
        key=lambda x: (-x[1], x[0]),
    )
    rest = sorted(
        ((n, s) for n, s in named_numels if not is_embedding_name(n)),
        key=lambda x: (-x[1], x[0]),
    )
    if not embed and not rest:
        raise ValueError("no tensors to fragment")

    fragments: list[Fragment] = []
    if embed:
        fragments.append(Fragment(MERGE_AVG, embed))

    n_bins = max(1, min(num_fragments - len(fragments), len(rest))) if rest else 0
    bins = [Fragment(MERGE_RDA) for _ in range(n_bins)]
    for name, numel in rest:
        target = min(bins, key=lambda b: b.numel)  # min() is stable: first-smallest wins
        target.tensors.append((name, numel))
    fragments.extend(bins)
    return FragmentLayout(fragments)
