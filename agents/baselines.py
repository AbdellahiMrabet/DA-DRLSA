# agents/baselines.py - non-learning schedulers; each returns a bid vector (or None = kube default)
import numpy as np


def _onehot_bids(n: int, idx: int) -> np.ndarray:
    b = -np.ones(n, dtype=np.float32)
    b[idx] = 1.0
    return b


class RoundRobin:
    name = "round_robin"
    def __init__(self): self.i = -1
    def reset(self): self.i = -1
    def act(self, env, obs):
        n = env.N
        for k in range(1, n + 1):
            j = (self.i + k) % n
            if env.mask[j]:
                self.i = j
                return _onehot_bids(n, j)
        return _onehot_bids(n, 0)             # nothing feasible -> env delays anyway


class LeastLoaded:
    name = "least_loaded"
    def reset(self): pass
    def act(self, env, obs):
        load = np.where(env.mask, env.cpu_alloc / env.cores, np.inf)
        return _onehot_bids(env.N, int(np.argmin(load)))


class FastestFeasible:
    """Timing-aware heuristic: smallest pWCET among feasible nodes (EDF-like with feasibility check)."""
    name = "fastest_feasible"
    def reset(self): pass
    def act(self, env, obs):
        t = np.where(env.mask, env.pwcet, np.inf)
        return _onehot_bids(env.N, int(np.argmin(t)))


class KubeDefault:
    name = "kube_default"
    def reset(self): pass
    def act(self, env, obs):
        return None


BASELINES = {c.name: c for c in (RoundRobin, LeastLoaded, FastestFeasible, KubeDefault)}
