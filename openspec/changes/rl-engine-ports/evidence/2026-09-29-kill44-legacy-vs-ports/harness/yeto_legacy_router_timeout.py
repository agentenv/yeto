"""Legacy-baseline environment shim (NOT part of yeto): the substitute base image's
megatron-bridge import takes ~25s, exceeding Miles' hard-coded 30s router readiness
wait. Raise only that wait (router_manager.start_router) to >=300s."""
import importlib.abc, importlib.machinery, sys
_T = "miles.utils.http_utils"
class _Loader(importlib.abc.Loader):
    def __init__(self, inner): self.inner = inner
    def create_module(self, spec): return self.inner.create_module(spec)
    def exec_module(self, module):
        self.inner.exec_module(module)
        orig = module.wait_for_server_ready
        def wait_for_server_ready(*a, timeout=30, **k):
            return orig(*a, timeout=max(timeout, 300), **k)
        module.wait_for_server_ready = wait_for_server_ready
class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name != _T: return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec and spec.loader: spec.loader = _Loader(spec.loader)
        return spec
sys.meta_path.insert(0, _Finder())
