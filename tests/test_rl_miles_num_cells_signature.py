import sys, types
from yeto.rl.adapters.miles import config as mc

def test_call_num_cells_handles_both_pin_signatures(monkeypatch):
    seen = {}
    def ports_v1(config, *, role):
        seen["v1"] = role; return 1
    assert mc._call_num_cells(ports_v1, object(), "actor") == 1 and seen["v1"] == "actor"
    fake = types.ModuleType("miles.utils.args.trainer_utils")
    fake.compute_trainer_total_gpus = lambda config, *, role: 8
    monkeypatch.setitem(sys.modules, "miles", types.ModuleType("miles"))
    monkeypatch.setitem(sys.modules, "miles.utils", types.ModuleType("miles.utils"))
    monkeypatch.setitem(sys.modules, "miles.utils.args", types.ModuleType("miles.utils.args"))
    monkeypatch.setitem(sys.modules, "miles.utils.args.trainer_utils", fake)
    def k31(args, *, total_gpus):
        seen["gpus"] = total_gpus; return 1
    assert mc._call_num_cells(k31, object(), "actor") == 1 and seen["gpus"] == 8
