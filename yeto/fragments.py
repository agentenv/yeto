"""Balanced tensor fragmentation.

Two patterns partition the trainable tensors into P fragments:

* ``binpack`` — greedy number partitioning: iterate over tensors in
  descending size order, placing each into the fragment with the smallest
  running total. Balances fragment sizes exactly but groups tensors
  arbitrarily with respect to network depth.
* ``strided`` — transformer layer i goes to fragment i mod P, so each
  fragment interleaves blocks from across the network's depth (the
  Streaming DiLoCo finding: strided fragments train slightly better than
  depth-contiguous ones). Tensors with no layer index are bin-packed onto
  the strided fragments afterwards.

Either way, embedding-like tensors are isolated into their own fragment
merged with direct averaging (embedding outer gradients lack the
near-orthogonality that motivates RDA); all other fragments use
radial-directional averaging.

The layout must be identical on every learner, so packing is deterministic:
ties break on tensor name and fragment index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MERGE_AVG = 0
MERGE_RDA = 1

_EMBED_MARKERS = ("embed", "wte", "wpe", "lm_head", "shared.weight")

# Transformer block index in a parameter FQN: "...layers.12...", "...h.3...",
# "...blocks.7...". Used by the strided pattern only.
_LAYER_RE = re.compile(r"\.(?:layers|h|blocks)\.(\d+)\.")

FRAGMENT_PATTERNS = ("binpack", "strided")


def is_embedding_name(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in _EMBED_MARKERS)


def layer_index(name: str) -> int | None:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


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


def build_layout(
    named_numels: list[tuple[str, int]],
    num_fragments: int,
    pattern: str = "binpack",
    *,
    avg_name_regex: str | None = None,
) -> FragmentLayout:
    """Partition (name, numel) pairs into at most `num_fragments` fragments.

    Embedding-like tensors share one avg-merged fragment; callers may add a
    regex for other vector-like tensors (for example diffusion bias/norm
    tensors). The rest go into the remaining fragments with RDA, placed by
    ``pattern`` (see module docstring). If there are fewer non-embedding
    tensors than bins, the fragment count shrinks.
    """
    if num_fragments < 1:
        raise ValueError("num_fragments must be >= 1")
    if pattern not in FRAGMENT_PATTERNS:
        raise ValueError(f"pattern must be one of {FRAGMENT_PATTERNS}, got {pattern!r}")
    avg_rx = re.compile(avg_name_regex) if avg_name_regex else None

    def use_avg(name: str) -> bool:
        return is_embedding_name(name) or (avg_rx is not None and bool(avg_rx.search(name)))

    avg = sorted(
        ((n, s) for n, s in named_numels if use_avg(n)),
        key=lambda x: (-x[1], x[0]),
    )
    rest = sorted(
        ((n, s) for n, s in named_numels if not use_avg(n)),
        key=lambda x: (-x[1], x[0]),
    )
    if not avg and not rest:
        raise ValueError("no tensors to fragment")

    fragments: list[Fragment] = []
    if avg:
        fragments.append(Fragment(MERGE_AVG, avg))

    n_bins = max(1, min(num_fragments - len(fragments), len(rest))) if rest else 0
    bins = [Fragment(MERGE_RDA) for _ in range(n_bins)]
    if pattern == "strided" and n_bins > 0:
        layered = [(n, s) for n, s in rest if layer_index(n) is not None]
        loose = [(n, s) for n, s in rest if layer_index(n) is None]
        # Layer i -> bin i mod n_bins; distinct layers spread before reuse.
        distinct = sorted({layer_index(n) for n, _ in layered})
        bin_of_layer = {layer: i % n_bins for i, layer in enumerate(distinct)}
        for name, numel in sorted(layered, key=lambda x: (layer_index(x[0]), x[0])):
            bins[bin_of_layer[layer_index(name)]].tensors.append((name, numel))
        rest = loose  # norms, biases, etc. fall through to bin-packing below
    for name, numel in rest:
        target = min(bins, key=lambda b: b.numel)  # min() is stable: first-smallest wins
        target.tensors.append((name, numel))
    fragments.extend(b for b in bins if b.tensors)  # strided can leave bins empty
    return FragmentLayout(fragments)
