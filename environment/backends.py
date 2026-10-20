# environment/backends.py - two interchangeable execution backends
#   SimBackend : calibrated discrete-time simulator (fast, used for training)
#   K8sBackend : real pods on a k3d/minikube cluster (used for evaluation / fine-tuning)
import time
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from config import NODE_PROFILES, SIM_PARAMS, DRAIN_TIMEOUT_S


@dataclass
class Completion:
    tid: int
    node_idx: int
    latency: float            # submit -> finished (s)
    startup: Optional[float]
    exec_s: Optional[float]
    failed: bool = False


class SimBackend:
    def __init__(self, seed: int = 0, params: Optional[dict] = None):
        self.p = dict(SIM_PARAMS if params is None else params)
        self.nodes = NODE_PROFILES
        self.reset(seed)

    def reset(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.t = 0.0
        self.tasks = []                         # dicts, one per in-flight task

    def now(self) -> float:
        return self.t

    def node_ready(self) -> np.ndarray:
        return np.ones(len(self.nodes), dtype=bool)

    def _alloc(self, i: int) -> float:
        return sum(x["task"].cpu for x in self.tasks if x["node"] == i)

    def place(self, task, node_idx: Optional[int]) -> int:
        if node_idx is None:                    # emulate default scheduler: least allocated
            node_idx = int(np.argmin([self._alloc(i) / n["cores"] for i, n in enumerate(self.nodes)]))
        util = min(self._alloc(node_idx) / self.nodes[node_idx]["cores"], 1.0)
        med = self.p["startup_median_s"][node_idx]
        startup = med * self.rng.lognormal(0.0, self.p["startup_sigma"]) * (1 + self.p["startup_load_factor"] * util)
        work = task.work * self.rng.lognormal(0.0, self.p["exec_sigma"])
        if self.rng.random() < self.p["stall_prob"]:
            work *= self.rng.uniform(*self.p["stall_range"])
        self.tasks.append(dict(task=task, node=node_idx, t_submit=self.t, t_start=self.t + startup,
                               remaining=work, started=False, exec_begin=None))
        return node_idx

    def advance(self, dt: float) -> List[Completion]:
        out, step = [], self.p["dt"]
        t_end = self.t + dt
        while self.t < t_end - 1e-9:
            h = min(step, t_end - self.t)
            self.t += h
            for x in list(self.tasks):
                if not x["started"]:
                    if self.t >= x["t_start"]:
                        x["started"], x["exec_begin"] = True, x["t_start"]
                    else:
                        continue
                i = x["node"]
                util = min(self._alloc(i) / self.nodes[i]["cores"], 1.0)
                speed = x["task"].cpu / (1.0 + self.p["interference_kappa"] * util ** 2)
                x["remaining"] -= speed * h
                if x["remaining"] <= 0:
                    self.tasks.remove(x)
                    out.append(Completion(x["task"].tid, i, self.t - x["t_submit"],
                                          x["t_start"] - x["t_submit"], self.t - x["exec_begin"]))
        return out

    def drain(self) -> List[Completion]:
        out = []
        while self.tasks and self.t < 1e5:
            out += self.advance(0.5)
        return out


class K8sBackend:
    def __init__(self):
        from environment.pod_manager import InferenceTaskManager
        self.mgr = InferenceTaskManager()
        self.names = [n["name"] for n in NODE_PROFILES]
        self.t0 = time.time()
        self._pod_by_tid = {}

    def reset(self, seed=None):
        self.mgr.cleanup_all()
        self.t0 = time.time()
        self._pod_by_tid.clear()

    def now(self) -> float:
        return time.time() - self.t0

    def node_ready(self) -> np.ndarray:
        return self.mgr.node_ready()

    def place(self, task, node_idx: Optional[int]) -> int:
        name = None if node_idx is None else self.names[node_idx]
        ok, pod, _ = self.mgr.submit(task, name, -1 if node_idx is None else node_idx)
        if not ok:
            return -1
        self._pod_by_tid[task.tid] = pod
        return node_idx if node_idx is not None else self.mgr.resolve_binding(pod)

    def _to_completions(self, recs) -> List[Completion]:
        return [Completion(r["task"].tid, r["node_idx"], r["latency"], r["startup"], r["exec_s"], r["failed"])
                for r in recs]

    def advance(self, dt: float) -> List[Completion]:
        out, t_end = [], time.time() + dt
        while True:
            out += self._to_completions(self.mgr.poll())
            rest = t_end - time.time()
            if rest <= 0:
                return out
            time.sleep(min(0.25, rest))

    def drain(self) -> List[Completion]:
        out, t_end = [], time.time() + DRAIN_TIMEOUT_S
        while self.mgr.inflight and time.time() < t_end:
            out += self._to_completions(self.mgr.poll())
            time.sleep(0.25)
        return out
