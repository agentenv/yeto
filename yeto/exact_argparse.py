"""Argument parsers that take long options by their full name only.

argparse's default ``allow_abbrev=True`` resolves any unambiguous prefix of a
long option to that option, so a misspelled name can set a different option
without a word: ``yeto launch --rollout-num-gpus 8`` used to set
``--rollout-num-gpus-per-engine`` (the dedicated rollout GPUs are
``--rl-rollout-gpus``). :class:`ExactArgumentParser` turns abbreviations off
(sub-parsers inherit the class) and, when an option is not recognized, names
the options that were probably meant.

Short options are unaffected: ``allow_abbrev=False`` only stops prefix matching
of long options; combined single-dash flags (``-vx``) still parse.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence

MAX_SUGGESTIONS = 4


class ExactArgumentParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` with ``allow_abbrev=False`` and suggestions for
    an unrecognized option. ``add_subparsers`` makes its sub-parsers of this class."""

    def __init__(self, *args, allow_abbrev: bool = False, **kwargs):
        super().__init__(*args, allow_abbrev=allow_abbrev, **kwargs)

    # argparse's own parse_args stays the entry point (it calls parse_known_args, then
    # error() with "unrecognized arguments: ..."): only that message gains the hints.
    def parse_known_args(self, args=None, namespace=None):
        argv = list(sys.argv[1:] if args is None else args)
        parsed, extras = super().parse_known_args(argv, namespace)
        self._exact_last_parse = (argv, list(extras))
        return parsed, extras

    def error(self, message):
        argv, extras = getattr(self, "_exact_last_parse", ((), ()))
        if extras and message == argparse._("unrecognized arguments: %s") % " ".join(extras):
            message = unrecognized_message(self, argv, extras)
        super().error(message)


def _sub_parsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:  # noqa: SLF001 - argparse has no public lookup
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return action.choices
    return {}


def command_path(parser: argparse.ArgumentParser, argv: Sequence[str]) -> list[argparse.ArgumentParser]:
    """``parser`` and the sub-parsers ``argv`` selects, outermost first."""
    path = [parser]
    for token in argv:
        choices = _sub_parsers(path[-1])
        if not choices:
            break
        if token in choices:
            path.append(choices[token])
    return path


def suggestions(token: str, parsers: Sequence[argparse.ArgumentParser]) -> list[str]:
    """Options of ``parsers`` that ``token`` (an unrecognized option) probably meant:
    an option with the same destination, the options it is a prefix of (what an
    abbreviation used to pick), then the closest spellings (difflib)."""
    name = token.split("=", 1)[0]
    options: dict[str, argparse.Action] = {}
    for parser in parsers:
        options.update(parser._option_string_actions)  # noqa: SLF001
    long_options = [o for o in options if o.startswith("--")]
    dest = name.lstrip("-").replace("-", "_")
    same_dest = sorted({o for o in long_options if options[o].dest == dest})
    extends = sorted(o for o in long_options if o.startswith(name) and o != name)
    close = difflib.get_close_matches(name, long_options, n=MAX_SUGGESTIONS, cutoff=0.6)
    out: list[str] = []
    for option in (*same_dest, *extends, *close):
        if option not in out:
            out.append(option)
    return out[:MAX_SUGGESTIONS]


def unrecognized_message(parser: argparse.ArgumentParser, argv: Sequence[str],
                         extras: Sequence[str]) -> str:
    """argparse's ``unrecognized arguments: ...`` plus, for each unrecognized long
    option, the options it probably meant."""
    lines = [f"unrecognized arguments: {' '.join(extras)}"]
    parsers = command_path(parser, argv)
    for token in extras:
        if not token.startswith("--") or token == "--":
            continue
        hints = suggestions(token, parsers)
        name = token.split("=", 1)[0]
        if hints:
            lines.append(f"  {name}: options are taken by their full name; did you mean "
                         f"{', '.join(hints)}?")
    return "\n".join(lines)
