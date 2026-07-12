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
radial-directional averaging by default. ``matrix_merge="iso"`` switches
those matrix fragments to Iso-C-style isotropic aggregation (IsoLoCo,
arXiv 2607.03011): the syncer direct-averages the per-tensor deltas and
flattens the singular-value spectrum of each averaged matrix to its mean.
Iso needs each tensor's 2D shape (carried in HELLO); tensors that are not
2D matrices join the direct-averaged fragment, matching the paper's rule
that non-matrix parameters skip the isotropic correction.

The layout must be identical on every learner, so packing is deterministic:
ties break on tensor name and fragment index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MERGE_AVG = 0
MERGE_RDA = 1
MERGE_ISO = 2

# CLI value -> merge mode for the non-embedding (matrix) fragments.
MATRIX_MERGE_MODES = {"rda": MERGE_RDA, "iso": MERGE_ISO}

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
    # {name: (rows, cols)}, populated exactly for MERGE_ISO fragments (the
    # syncer needs the 2D view; avg/RDA operate on flat slices).
    shapes: dict[str, tuple[int, int]] | None = None

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
    matrix_merge: str = "rda",
    named_shapes: dict[str, tuple[int, ...]] | None = None,
) -> FragmentLayout:
    """Partition (name, numel) pairs into at most `num_fragments` fragments.

    Embedding-like tensors share one avg-merged fragment; callers may add a
    regex for other vector-like tensors (e.g. NAVA bias/norm/modulation).
    The rest go into the remaining fragments with ``matrix_merge`` (RDA by
    default, or Iso-C-style "iso"), placed by ``pattern`` (see module
    docstring). If there are fewer non-embedding tensors than bins, the
    fragment count shrinks. Iso requires ``named_shapes`` (full torch shapes
    keyed by name); tensors without a 2D shape fall back to the avg
    fragment.
    """
    if num_fragments < 1:
        raise ValueError("num_fragments must be >= 1")
    if pattern not in FRAGMENT_PATTERNS:
        raise ValueError(f"pattern must be one of {FRAGMENT_PATTERNS}, got {pattern!r}")
    if matrix_merge not in MATRIX_MERGE_MODES:
        raise ValueError(
            f"matrix_merge must be one of {tuple(MATRIX_MERGE_MODES)}, got {matrix_merge!r}"
        )
    matrix_mode = MATRIX_MERGE_MODES[matrix_merge]
    if matrix_mode == MERGE_ISO and named_shapes is None:
        raise ValueError('matrix_merge="iso" requires named_shapes')
    avg_rx = re.compile(avg_name_regex) if avg_name_regex else None

    def use_avg(name: str) -> bool:
        if is_embedding_name(name) or (avg_rx is not None and bool(avg_rx.search(name))):
            return True
        # Iso operates on 2D matrices only; the paper leaves non-matrix
        # parameters to plain averaging.
        return matrix_mode == MERGE_ISO and len(named_shapes.get(name, ())) != 2

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
    bins = [Fragment(matrix_mode) for _ in range(n_bins)]
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
    if matrix_mode == MERGE_ISO:
        for frag in fragments:
            if frag.merge_mode == MERGE_ISO:
                frag.shapes = {
                    name: (int(named_shapes[name][0]), int(named_shapes[name][1]))
                    for name, _ in frag.tensors
                }
    return FragmentLayout(fragments)
