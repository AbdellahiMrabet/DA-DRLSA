# environment/rt_env.py - deadline-aware multi-agent scheduling environment (adapted from K8sEnv)
#
# What changed w.r.t. the original K8sEnv:
#   * workload: finite inference tasks with hard/soft deadlines (was: nginx pods with TTL)
#   * reward : deadline misses + latency vs. deadline (was: API latency + load balance bonus)
#   * safety : action *mask* = resource fit AND pWCET <= remaining slack (was: HSP projection)
#   * agents : one agent per node; each outputs a bid in [-1,1], placement = argmax of masked bids
#   * state  : kept in the env's own ledger, no `kubectl describe` parsing
#   * backends: SimBackend (training) / K8sBackend (real pods) behind the same interface
import time
from collections import deque
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (NODE_PROFILES, NUM_NODES, CPU_CAP, MEM_CAP, MAX_RUNNING_NORM, MAX_CPU_REQ,
                    MAX_MEM_REQ, MAX_WORK, MAX_DEADLINE, MAX_PENDING, MAX_STEPS, MISS_COST,
                    LATENCY_WEIGHT, DELAY_PENALTY, REWARD_CLIP)
from environment.workload import TaskGenerator
from environment.backends import SimBackend, K8sBackend
from timing import PWCETEstimator
from metrics import MetricsTracker

NODE_F, TASK_F = 8, 6


class RTSchedEnv(gym.Env):
    """
    Per step the env must place ONE task (the pending task with the earliest absolute deadline).
    action : np.ndarray (N,) of bids in [-1, 1]  |  None -> delegate to the default kube-scheduler
    obs    : (N, obs_dim) one local observation per agent
    reward : scalar team reward (delayed: it contains the outcome of tasks that finished meanwhile)
    info   : 'state' (global state for the centralised critic), 'mask' (N,), 'metrics' at the end
    """

    def __init__(self, backend: str = "sim", seed: int = 0, max_steps: int = MAX_STEPS,
                 pwcet_method: str = "empirical"):
        super().__init__()
        self.N = NUM_NODES
        self.cores = np.array([n["cores"] for n in NODE_PROFILES])
        self.mem_cap = np.array([n["mem_mb"] for n in NODE_PROFILES], dtype=float)
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.backend_name = backend
        self.backend = SimBackend(seed) if backend == "sim" else K8sBackend()
        self.gen = TaskGenerator(self.rng)
        self.est = PWCETEstimator(pwcet_method)
        self.metrics = MetricsTracker()

        self.obs_dim = NODE_F + TASK_F + self.N + 2
        self.state_dim = self.N * NODE_F + TASK_F + 2
        self.observation_space = spaces.Box(-1.0, 2.0, (self.N, self.obs_dim), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (self.N,), dtype=np.float32)
        self._init_episode_state()

    # ------------------------------------------------------------------ helpers
    def _init_episode_state(self):
        self.cpu_alloc = np.zeros(self.N)
        self.mem_alloc = np.zeros(self.N)
        self.running = np.zeros(self.N)
        self.miss_ema = np.zeros(self.N)
        self.inflight = {}                    # tid -> (task, node_idx)
        self.pending = []                     # tasks waiting for a placement decision
        self.current = None
        self.step_count = 0
        self.mask = np.zeros(self.N, bool)
        self.res_ok = np.zeros(self.N, bool)
        self.pwcet = np.zeros(self.N)
        self.timing_relaxed = False

    def _arrival(self):
        self.pending.append(self.gen.sample(self.backend.now()))

    def _pop_next(self):
        self.pending.sort(key=lambda t: t.deadline_abs)
        return self.pending.pop(0)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        super().reset(seed=seed)
        self.backend.reset(seed)
        self.gen.reset(self.rng)
        self.est.reset()
        self.metrics.reset()
        self._init_episode_state()
        self._arrival()
        self.current = self._pop_next()
        obs, state = self._build(self.current)
        return obs, {"state": state, "mask": self.mask.copy(), "rate": self.gen.rate}

    def _build(self, task):
        now = self.backend.now()
        slack = max(task.deadline_abs - now, 0.0)
        ready = self.backend.node_ready()
        cpu_ok = self.cpu_alloc + task.cpu <= CPU_CAP * self.cores + 1e-9
        mem_ok = self.mem_alloc + task.mem_mb <= MEM_CAP * self.mem_cap + 1e-9
        self.res_ok = ready & cpu_ok & mem_ok
        self.pwcet = np.array([self.est.estimate(i, task.ttype, task.work, task.cpu) for i in range(self.N)])
        timing_ok = self.pwcet <= slack
        mask = self.res_ok & timing_ok
        self.timing_relaxed = (not mask.any()) and self.res_ok.any()
        if self.timing_relaxed:               # nobody is predicted to make it: best effort on feasible nodes
            mask = self.res_ok.copy()
        self.mask = mask

        ratio = np.clip(self.pwcet / max(slack, 0.05), 0.0, 2.0) / 2.0
        cpu_u = self.cpu_alloc / self.cores
        node = np.stack([cpu_u,
                         self.mem_alloc / self.mem_cap,
                         np.minimum(self.running / MAX_RUNNING_NORM, 1.0),
                         ratio,
                         ready.astype(float),
                         self.cores / self.cores.max(),
                         self.miss_ema,
                         mask.astype(float)], axis=1)                       # (N, 8)
        task_f = np.array([float(task.cls == "hard"), task.cpu / MAX_CPU_REQ, task.mem_mb / MAX_MEM_REQ,
                           min(slack / MAX_DEADLINE, 1.0), task.work / MAX_WORK,
                           min(len(self.pending) / MAX_PENDING, 1.0)])      # (6,)
        summ = np.array([cpu_u.mean(), cpu_u.std()])
        eye = np.eye(self.N)
        obs = np.concatenate([node, np.tile(task_f, (self.N, 1)), eye, np.tile(summ, (self.N, 1))], axis=1)
        state = np.concatenate([node.ravel(), task_f, summ])
        return obs.astype(np.float32), state.astype(np.float32)

    # ------------------------------------------------------------------ step
    def step(self, action):
        task = self.current
        reward, delayed, placed = 0.0, False, -1
        self.step_count += 1

        # ---- decision
        if action is None:                                        # default kube-scheduler baseline
            if self.res_ok.any():
                placed = self.backend.place(task, None)
        elif self.mask.any():
            bids = np.where(self.mask, np.asarray(action, dtype=float).reshape(self.N), -np.inf)
            placed = self.backend.place(task, int(np.argmax(bids)))
        if placed >= 0:
            self.inflight[task.tid] = (task, placed)
            self.cpu_alloc[placed] += task.cpu
            self.mem_alloc[placed] += task.mem_mb
            self.running[placed] += 1
        else:                                                     # strategic delay (or failed submit)
            delayed = True
            reward -= DELAY_PENALTY
            self.pending.append(task)

        # ---- time passes; a new task arrives
        reward += self._advance(self.gen.next_interarrival())
        self._arrival()

        # ---- drop tasks that can no longer be served, cap queue
        reward += self._drop_expired()

        truncated = self.step_count >= self.max_steps
        if truncated:                                             # account for outcomes of in-flight tasks
            reward += self._settle(self.backend.drain())
            for t in self.pending:
                reward += self._record_drop(t)
            self.pending = []
        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        self.metrics.add_step(reward, delayed)

        info = {"delayed": delayed, "timing_relaxed": self.timing_relaxed}
        if truncated:
            self.current = None
            obs = np.zeros((self.N, self.obs_dim), np.float32)
            info.update(state=np.zeros(self.state_dim, np.float32), mask=np.zeros(self.N, bool),
                        metrics=self.metrics.summary())
            return obs, reward, False, True, info
        if not self.pending:
            self._arrival()
        self.current = self._pop_next()
        obs, state = self._build(self.current)
        info.update(state=state, mask=self.mask.copy())
        return obs, reward, False, False, info

    # ------------------------------------------------------------------ bookkeeping
    def _advance(self, dt: float) -> float:
        return self._settle(self.backend.advance(dt))

    def _settle(self, completions) -> float:
        r = 0.0
        for c in completions:
            if c.tid not in self.inflight:
                continue
            task, node = self.inflight.pop(c.tid)
            n = c.node_idx if c.node_idx >= 0 else node
            if node >= 0:
                self.cpu_alloc[node] = max(self.cpu_alloc[node] - task.cpu, 0.0)
                self.mem_alloc[node] = max(self.mem_alloc[node] - task.mem_mb, 0.0)
                self.running[node] = max(self.running[node] - 1, 0)
            missed = c.failed or c.latency > task.deadline
            if not c.failed and n >= 0:
                self.est.update(n, task.ttype, c.latency)
            if n >= 0:
                self.miss_ema[n] = 0.9 * self.miss_ema[n] + 0.1 * float(missed)
            if missed:
                r -= MISS_COST[task.cls]
            else:
                r -= LATENCY_WEIGHT * c.latency / task.deadline
            self.metrics.add_completion(task.cls, c.latency if not c.failed else float("nan"), missed)
        return r

    def _record_drop(self, task) -> float:
        self.metrics.add_completion(task.cls, float("nan"), True, dropped=True)
        return -MISS_COST[task.cls]

    def _drop_expired(self) -> float:
        now, r, keep = self.backend.now(), 0.0, []
        for t in self.pending:
            if t.deadline_abs < now:
                r += self._record_drop(t)
            else:
                keep.append(t)
        self.pending = keep
        while len(self.pending) > MAX_PENDING:                    # overload protection
            self.pending.sort(key=lambda t: t.deadline_abs)
            r += self._record_drop(self.pending.pop(0))
        return r

    def close(self):
        if self.backend_name == "k8s":
            self.backend.mgr.cleanup_all()
