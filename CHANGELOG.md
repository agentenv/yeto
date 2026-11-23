# Changelog

## Unreleased (0.2.0)

### Breakers: one registry, one config section

Every automatic stop, drop, pause or refusal is now a registered breaker with
one name, `<layer>.<entry>`. Reference: [docs/BREAKERS.md](docs/BREAKERS.md).
How to use it: [docs/KIMI_K31_ASYNC_RL_GUIDE.md, section 7](docs/KIMI_K31_ASYNC_RL_GUIDE.md#7-breakers-and-cost-protection).

Added:

- A `breakers:` section in the run config. It sets any breaker of the seven
  layers (`run`, `launch`, `trainer`, `sampler`, `harness`, `pool`,
  `monitor`), per pool with `breakers.pools.<name>`. `yeto launch --breakers
  FILE` applies the `launch` and `trainer` layers without `run up`. A sampler
  spec may carry its own `breakers` key for the `sampler` and `harness` layers.
- `yeto breaker show --config run.json`: every breaker's effective value and
  where it came from. `yeto breaker list --markdown`: the registry.
  `yeto breaker check` is the new name of the health check.
- One budget: `breakers.run.budget.usd` is inherited by `launch.budget` and
  `pool.spend_cap` when set in the section.
- Dependency rules D1 to D21 in the `run up` preflight, `yeto launch --dry-run`,
  the pool dry run and `show`. D2, D3, D6, D8, D15, D16 and D21 refuse the
  start; the others warn. `breakers.deps.outage_tolerance_s` and
  `breakers.deps.overrides`.
- `rl_breaker` events at every trip site and `rl_breaker_status` snapshots,
  on each component's existing tape. `run up` writes `events.jsonl` in its
  state directory.
- `stop_reason` (`breaker:<name>`) on the launcher's last line, the run record
  and the `run up` `state.json`. Exit codes do not change.
- A Breakers panel on the dashboard, and a `breakers` object in
  `yeto run status`.
- Two native breakers, on by default as alerts only:
  - `run.release_stall`: no new policy release for a while after the first
    one. `action: stop` stops the trainer.
  - `sampler.infra_streak`: agentic groups in a row lost to infrastructure.
    `action: pause` stops claiming new tasks until a group is written or a
    cool-down passes.
- New knobs that used to be constants or env only: sampler spec
  `max_read_outage_s` and `max_damaged_latest_reads`, `--nc-lease-overdue-minutes`,
  `--budget-usd off`, `harness.sandbox_ttl.limit_s`.

Unchanged:

- A config without `breakers:` resolves to exactly the values it had before. A
  golden test (`tests/golden/breakers`) locks this for every example config.
- Data-correctness checks (`kind: invariant`) cannot be disabled.
- `--no-cost-guard` still turns the cost guard off. It is now the `cost_guard`
  switch group (`launch.budget`, `launch.idle`, `launch.nc_lease_overdue`).

Deprecated (removal in 0.4.0):

- Legacy spellings of a breaker setting are now aliases of a registry entry.
  Those that duplicate another place, or that only an environment variable
  could set, are deprecated: for example `HARBOR_SETUP_RETRIES`,
  `HARBOR_MAX_SANDBOXES`, the `YETO_AGENTENV_*` and `YETO_BOAT_*` caps,
  `YETO_AGENTENV_SKIP_REACH_CHECK` (now `harness.reach_check.enabled: false`).
  The full list is the entries marked "(deprecated)" in
  [docs/BREAKERS.md](docs/BREAKERS.md).
- They keep working and print a note when used (`yeto run up`, `yeto launch`,
  the pool CLI, `yeto breaker show`). They will be removed two minor releases
  later, in 0.4.0. Move the settings into `breakers:`.
- Not deprecated: the native keys of a standalone sampler spec
  (`max_backlog_groups`, `agentic.*`, `prompts.dispense.*`,
  `reload_max_defer_s`, ...) are its canonical form. The `yeto launch` flags
  without a "(deprecated)" mark stay as well.
- A legacy knob that disagrees with the `breakers:` section is refused, with
  both locations named.
