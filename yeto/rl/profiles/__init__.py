"""Pinned single-command launch profiles for upstream Miles recipes.

A profile is a frozen description (model identity, checkpoint paths,
parallel layout, LoRA layout, SGLang serving flags) plus pure functions that
render the exact commands the matching ``scripts/*.sh`` launcher runs.  The
renderers never touch the network or a GPU so they can be snapshot-tested on
CPU; the shell launchers call them through ``python3 -m yeto.rl.profiles.<name>``
so a dry-run prints byte-identical commands to a real launch.
"""
