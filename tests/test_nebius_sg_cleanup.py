"""yeto down deletes a Nebius run's leaked sky security groups (S19 real-spot A2:
quota 128 hit by 125 leaked groups)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from yeto import cli
from yeto.cloud import nebius_sg_cleanup as sc


class FakeUtils:
    def __init__(self, groups, instances, stuck=()):
        self.groups = dict(groups)            # id -> name
        self.instances = instances
        self.stuck = set(stuck)
        self.deleted = []
        page = SimpleNamespace(items=[SimpleNamespace(metadata=SimpleNamespace(id=i, name=n))
                                      for i, n in self.groups.items()], next_page_token="")
        svc = SimpleNamespace(list=lambda req: page)
        vpc = SimpleNamespace(SecurityGroupServiceClient=lambda sdk: svc,
                              ListSecurityGroupsRequest=lambda **kw: kw)
        self.nebius = SimpleNamespace(vpc=lambda: vpc, sdk=lambda: None, sync_call=lambda x: x)

    def get_project_by_region(self, region):
        return "project-x"

    def list_instances(self, project_id):
        return self.instances

    def delete_security_group(self, gid):
        self.deleted.append(gid)
        if gid not in self.stuck:
            self.groups.pop(gid)

    def get_security_group_by_name(self, project_id, name):
        return next((i for i, n in self.groups.items() if n == name), None)


def test_deletes_only_this_runs_unused_groups():
    u = FakeUtils({"g1": "sky-sg-r1-l0-eu-north1-abc", "g2": "sky-sg-r1-head-abc",
                   "g3": "sky-sg-r1-l1-eu-north1-abc", "g4": "sky-sg-r10-l0-x",
                   "g5": "sky-sg-other-l0-x", "g6": "default-security-group"},
                  {"i1": {"security_group_ids": ["g3"]}})
    logs = []
    assert sc.delete_run_security_groups("r1", utils=u, log=logs.append) == ["g1", "g2"]
    assert u.deleted == ["g1", "g2"]  # g3 in use, g4/g5/g6 other names


def test_a_group_sky_could_not_delete_is_reported_not_counted():
    u = FakeUtils({"g1": "sky-sg-r1-l0"}, {}, stuck={"g1"})
    logs = []
    assert sc.delete_run_security_groups("r1", utils=u, log=logs.append) == []
    assert "NOT deleted" in logs[-1]


def test_never_raises_when_the_cloud_cannot_be_read():
    class Broken:
        def get_project_by_region(self, region):
            raise RuntimeError("no creds")

    logs = []
    assert sc.delete_run_security_groups("r1", utils=Broken(), log=logs.append) == []
    assert "not checked" in logs[-1]


@pytest.mark.real_nebius_sg_cleanup
def test_down_runs_cleanup_only_for_nebius_runs(monkeypatch):
    seen = []
    monkeypatch.setattr(sc, "delete_run_security_groups", lambda prefix: seen.append(prefix) or [])
    cli._nebius_sg_cleanup("r1", {"args": {"gpu": "nebius:1xh100@eu-north1", "cluster_prefix": "p1"}})
    cli._nebius_sg_cleanup("r2", {"args": {"gpu": "modal:1xh100", "syncer_region": "nebius/eu-north1"}})
    cli._nebius_sg_cleanup("r3", {"args": {"gpu": "modal:1xh100", "syncer_region": "us-east-1"}})
    assert seen == ["p1", "r2"]
