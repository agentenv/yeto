"""Read-only RL fleet dashboard (openspec change ``yeto-fleet-dashboard``).

One reducer (``reducer.Reducer``) folds learner tapes, the syncer tape, the
controller journal and the head's ``fleet.jsonl`` into JSON views; ``serve``
exposes them over a loopback-only GET-only HTTP server and ``export`` inlines
them into a single self-contained HTML file.
"""
