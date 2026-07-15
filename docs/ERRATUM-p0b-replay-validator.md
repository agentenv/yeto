# P0b post-deletion replay validator erratum

This erratum authorizes one replay-only source transition for the sealed P0b
acquisition whose frozen source commit is
`8d58208cacafef12cb95f2642b4fa700531151b4`.

The acquisition, scientific commands, model/data inputs, optimizer states,
candidate deltas, event tapes, learner traces, evaluation rows, per-example
losses, result rows, P0a parent, deletion evidence, and immutable cloud object
generations remain unchanged. The descendant changes only controller-side
validation and provenance.

## Authorized corrections

1. When comparing a P0b result with its exact P0a parent, accept the legacy
   P0a `work` mapping only when the P0b mapping differs solely by the derived
   positive-work facts `learner_count=4` and
   `learner_steps_per_learner=128`. Any missing, different, or additional work
   field remains an error.
2. Apply the lifecycle ordering already used by
   `finalize_p0_lifecycle.py`, `validate_p0_replay.py`, and the existing
   same-second regression: provisioning start and completion are strictly
   ordered, the immutable artifact seal may equal the whole-second deletion
   request timestamp, and deletion completion remains strictly later.
3. Permit the post-deletion replay validator to run from a clean descendant
   of the frozen P0b commit only when this exact erratum is present and
   hash-bound in the replay report. A P1-R0 parent gate must verify the same
   source-rebind provenance before accepting that report.

## Required verification

The descendant must add regression coverage for the exact legacy/new work
shape, reject incorrect learner counts or step counts, accept an equal
whole-second seal/request boundary, reject reversed lifecycle ordering, pass
the complete repository test suite, and then replay every immutable captured
step after verified exact-ID VM and boot-disk deletion.
