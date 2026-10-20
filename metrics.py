# metrics.py - real-time oriented metrics (misses and tail latency, not just means)
import json
import numpy as np


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rows = []            # (cls, latency, missed, dropped)
        self.rewards = []
        self.delays = 0
        self.decision_ms = []

    def add_completion(self, cls: str, latency: float, missed: bool, dropped: bool = False):
        self.rows.append((cls, latency, missed, dropped))

    def add_step(self, reward: float, delayed: bool):
        self.rewards.append(reward)
        self.delays += int(delayed)

    def add_decision_time(self, ms: float):
        self.decision_ms.append(ms)

    def summary(self) -> dict:
        def block(sel):
            if not sel:
                return dict(n=0, miss_ratio=float("nan"))
            lat = np.array([r[1] for r in sel if not r[3] and np.isfinite(r[1])])
            out = dict(n=len(sel), miss_ratio=float(np.mean([r[2] for r in sel])))
            if len(lat):
                out.update(p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
                           p99=float(np.percentile(lat, 99)), max=float(lat.max()))
            return out
        d = np.array(self.decision_ms) if self.decision_ms else np.zeros(1)
        return dict(all=block(self.rows),
                    hard=block([r for r in self.rows if r[0] == "hard"]),
                    soft=block([r for r in self.rows if r[0] == "soft"]),
                    dropped=int(sum(r[3] for r in self.rows)),
                    delays=self.delays,
                    mean_step_reward=float(np.mean(self.rewards)) if self.rewards else 0.0,
                    decision_ms=dict(p50=float(np.percentile(d, 50)), p99=float(np.percentile(d, 99)),
                                     max=float(d.max())))

    def dump(self, path: str):
        with open(path, "w") as f:
            json.dump(self.summary(), f, indent=2)
