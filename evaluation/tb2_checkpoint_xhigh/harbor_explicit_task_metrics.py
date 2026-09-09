"""Isolated Harbor 0.18.0 metric repair for explicitly pinned task sources.

The upstream initializer binds dataset metrics only for config.datasets. This
job uses exact GitTaskIds in config.tasks; their source otherwise resolves to
an empty list and reporting cancels remaining trials with IndexError.
"""
from __future__ import annotations


def install():
    from harbor.job import Job
    from harbor.metrics.factory import MetricFactory
    from harbor.metrics.mean import Mean

    original = Job._resolve_metrics

    async def resolve(config, task_configs):
        metrics = await original(config, task_configs)
        for source in {task.source or "adhoc" for task in task_configs}:
            if not metrics[source]:
                metrics[source] = [
                    MetricFactory.create_metric(metric.type, **metric.kwargs)
                    for metric in config.metrics
                ] or [Mean()]
        return metrics

    Job._resolve_metrics = staticmethod(resolve)


if __name__ == "__main__":
    install()
    from harbor.cli.main import app
    app()
