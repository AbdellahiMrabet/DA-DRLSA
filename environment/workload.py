# environment/workload.py - task model and arrival process
from dataclasses import dataclass
import numpy as np
from config import TASK_TYPES, ARRIVAL_RATE_RANGE, BURST_PROB, BURST_FACTOR


@dataclass
class Task:
    tid: int
    ttype: str
    cls: str            # "hard" | "soft"
    cpu: float          # cores (request == limit)
    mem_mb: int
    work: float         # core-seconds
    deadline: float     # relative deadline (s)
    arrival: float      # env time of arrival (s)

    @property
    def deadline_abs(self) -> float:
        return self.arrival + self.deadline


class TaskGenerator:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.names = list(TASK_TYPES)
        p = np.array([TASK_TYPES[n]["prob"] for n in self.names], dtype=float)
        self.p = p / p.sum()
        self.rate = 1.0
        self._tid = 0

    def reset(self, rng: np.random.Generator):
        self.rng = rng
        self.rate = float(rng.uniform(*ARRIVAL_RATE_RANGE))
        self._tid = 0

    def next_interarrival(self) -> float:
        rate = self.rate * (BURST_FACTOR if self.rng.random() < BURST_PROB else 1.0)
        return float(self.rng.exponential(1.0 / rate))

    def sample(self, now: float) -> Task:
        name = str(self.rng.choice(self.names, p=self.p))
        t = TASK_TYPES[name]
        self._tid += 1
        return Task(self._tid, name, t["cls"], t["cpu"], t["mem_mb"], t["work"], t["deadline"], now)
