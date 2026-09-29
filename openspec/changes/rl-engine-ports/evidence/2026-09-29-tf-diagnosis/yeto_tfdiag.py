"""DIAGNOSTIC ONLY (tf-diagnosis 2026-09-29); not part of yeto. Passive recorder of Miles
policy_loss_function inputs/outputs. Active only when /work/tfdiag.on exists."""
import os, sys, importlib.abc, importlib.util, itertools
TARGET = "miles.backends.training_utils.loss_hub.losses"
_cnt = itertools.count()
_last = {}

def _cpu(x):
    import torch
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu").clone()
    if isinstance(x, (list, tuple)):
        return [_cpu(v) for v in x]
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    return repr(type(x))

def _patch(mod):
    import torch
    orig_lp, orig_pl = mod.get_log_probs_and_entropy, mod.policy_loss_function
    def lp(*a, **kw):
        out = orig_lp(*a, **kw)
        _last["train_log_probs"] = [t.detach().float().cpu().clone() for t in out["log_probs"]]
        return out
    def pl(args, batch, logits, sum_of_sample_mean):
        _last.clear()
        loss, metrics = orig_pl(args, batch, logits, sum_of_sample_mean)
        try:
            d = getattr(args, "yeto_rl_grad_audit_dir", None) or "/work/tfdiag-noaudit"
            d = os.path.join(os.path.dirname(d.rstrip("/")), "tfdiag")
            os.makedirs(d, exist_ok=True)
            rec = {k: _cpu(v) for k, v in batch.items()}
            rec["train_log_probs"] = _last.get("train_log_probs")
            rec["loss"] = float(loss.detach().float().item())
            rec["metrics"] = {k: _cpu(v) for k, v in (metrics or {}).items()}
            rec["logits_dtype"] = str(logits.dtype); rec["logits_shape"] = list(logits.shape)
            rec["pid"] = os.getpid()
            torch.save(rec, os.path.join(d, f"mb-{os.getpid()}-{next(_cnt):05d}.tfd"))
        except Exception as e:  # never break training
            print("TFDIAG_ERR", repr(e), file=sys.stderr)
        return loss, metrics
    mod.get_log_probs_and_entropy, mod.policy_loss_function = lp, pl
    print("TFDIAG_PATCHED", os.getpid(), file=sys.stderr)

class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None:
            return None
        loader = spec.loader; orig = loader.exec_module
        def exec_module(m):
            orig(m); _patch(m)
        loader.exec_module = exec_module
        return spec

if os.path.exists("/work/tfdiag.on"):
    sys.meta_path.insert(0, _Finder())
