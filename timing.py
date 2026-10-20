# timing.py - online pWCET (probabilistic worst-case) estimation of end-to-end latency
from collections import defaultdict, deque
import numpy as np
from config import (PWCET_QUANTILE, PWCET_WINDOW, PWCET_MIN_SAMPLES, PWCET_PRIOR_MARGIN,
                    STARTUP_PRIOR_S, NODE_PROFILES)

try:
    from scipy.stats import genpareto
    _HAVE_SCIPY = True
except Exception:  # scipy optional
    _HAVE_SCIPY = False


def evt_quantile(x: np.ndarray, q_target: float = 0.999, u_q: float = 0.90) -> float:
    """Peaks-over-threshold: fit a GPD to the tail and extrapolate the q_target quantile."""
    x = np.asarray(x, dtype=float)
    if not _HAVE_SCIPY or len(x) < 100:
        return float(np.quantile(x, min(q_target, 0.99)))
    u = np.quantile(x, u_q)
    exc = x[x > u] - u
    if len(exc) < 20:
        return float(np.quantile(x, min(q_target, 0.99)))
    try:
        c, _, scale = genpareto.fit(exc, floc=0.0)
        p_exc = len(exc) / len(x)
        pr = (1.0 - q_target) / p_exc
        if pr >= 1.0:
            return float(np.quantile(x, q_target))
        return float(u + genpareto.isf(pr, c, loc=0.0, scale=scale))
    except Exception:
        return float(np.quantile(x, min(q_target, 0.99)))


class PWCETEstimator:
    """Per (node, task-type) high-quantile of observed end-to-end latency (submit -> done)."""

    def __init__(self, method: str = "empirical", quantile: float = PWCET_QUANTILE):
        assert method in ("empirical", "evt")
        self.method, self.q = method, quantile
        self.samples = defaultdict(lambda: deque(maxlen=PWCET_WINDOW))

    def reset(self):
        self.samples.clear()

    def update(self, node_idx: int, ttype: str, latency_s: float):
        self.samples[(node_idx, ttype)].append(latency_s)

    @staticmethod
    def prior(node_idx: int, work: float, cpu: float) -> float:
        cores = NODE_PROFILES[node_idx]["cores"]
        return STARTUP_PRIOR_S + PWCET_PRIOR_MARGIN * work / min(cpu, cores)

    def estimate(self, node_idx: int, ttype: str, work: float, cpu: float) -> float:
        s = self.samples.get((node_idx, ttype))
        prior = self.prior(node_idx, work, cpu)
        if s is None or len(s) < PWCET_MIN_SAMPLES:
            return prior
        arr = np.fromiter(s, dtype=float)
        if self.method == "evt":
            return evt_quantile(arr, q_target=max(self.q, 0.999))
        return float(np.quantile(arr, self.q))
