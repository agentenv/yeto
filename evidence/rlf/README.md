# RLF bridge evidence

The trusted-side bridge requires the rlenvfactory package shape (`public/`, `task.yaml`, `MANIFEST.json`) and an explicit runtime quality gate. AgentFlow node output is deliberately rejected as identity or acceptance evidence. Episodes carry task/trajectory/episode identities, clear secrets during idempotent teardown, and preserve signed grader evidence in neutral reward metadata.

The RL gate rejects empty, constant-reward, zero-advantage or non-updating rollouts. A healthy fixture must show nonzero reward variance, success count, absolute advantage sum, and a policy update.
