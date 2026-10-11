# Source audit — 2026-10-10

RLF revision: `9fb2bda9333d584385d53e126f3a7145e5392d7e`; its AgentFlow pin: `eb51a2e7a5c7d65da5718c4a41843098b60c80e0`.

## Corrections to earlier discussion

- `factory/paths.py:RepositoryLayout.task` puts full published packages at `tasks/ctf/<task_id>/`, including `task.json`. `runs/ctf/tasks/<task_id>/<run_id>/` contains attempt records, not the package.
- `runtime/shared/export.py` exports public handouts and manifests only. This export omits the full Docker build context and host receipt. It cannot alone provision an equivalent training environment; do not require this optional export as the sole admission input.
- `runtime/shared/package_validation.py:validate_task_package` returns `ValidatedTask` with no runtime evidence when static-only mode is enabled. Thus `validation.status=validated` alone does not prove a reference solve. Admission must verify runtime gate evidence bound to the current full package and explicitly handle missing evidence.
- `runtime/shared/agent_runtime.py:AgentRuntime.run` constructs a one-node Graph with concurrency=1 per invocation. RLF's outer CTF generation loop and repair/QA flow are Python host orchestration. AgentFlow supports DAG fanout generally, but this inspected integration does not demonstrate DAG-parallel RL rollouts or a throughput advantage over Yeto.
- AgentFlow already participates in RLF execution, including ordinary Solver attempts. Restricting its use to generation in the proposed Yeto integration is an architectural recommendation, not an upstream restriction or a user-approved prohibition on future online integration.

## Open design work

Reconcile proposal/spec/design/tasks with the full-package + host-receipt admission boundary before implementation. Preserve generation and runtime acceptance in the end-to-end goal. Compare an optional AgentFlow scheduler around Yeto's sampling/trajectory boundary separately from direct replacement: generic scheduling can coexist with RL-specific token capture, but correctness and throughput require testing, not inference from DAG support.
