"""yeto RL capability evaluation suite (S25 T-EVAL).

One repeatable GPU evaluation per capability and backend, one CLI, one verdict
format and one report:

    python -m tools.eval list
    python -m tools.eval run <E1..E5> --backend miles|verl --run-id <id> [--dry-run]
    python -m tools.eval judge <run dir>
    python -m tools.eval report <run dir> ... --out <report.md>

* :mod:`.cells` -- the capability x backend matrix (what each cell runs, its
  configuration, its judge, and whether the backend supports it);
* :mod:`.verdict` -- the verdict written by every judge (``verdict.json``);
* :mod:`.judges` -- the pre-registered criteria of each capability;
* :mod:`.modal_verl` -- verl cells: a direct Modal harness (syncer, engine sampler
  island, verl islands), derived from the S25 C2/C3V/C4 harnesses;
* ``miles_launch.sh`` -- Miles cells: the yeto launcher on Modal, derived from
  the S25 C1/C3 and S24 L5.4 run scripts;
* :mod:`.report` -- Markdown and HTML report over a set of verdicts.

Every run writes its raw data under ``<out>/<run id>/raw/`` before judging.
"""
