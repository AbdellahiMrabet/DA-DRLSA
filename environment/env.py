# env.py - MATD3 Deadline-Aware Kubernetes Pod Offloading Environment
import os
import time
import json
import traceback
from typing import Optional, Tuple, Dict, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import (
    NODE_PROFILES, NUM_NODES, CPU_CAP, MEM_CAP, MAX_RUNNING_NORM,
    MAX_CPU_REQ, MAX_MEM_REQ, MAX_WORK, MAX_DEADLINE, MAX_PENDING,
    MAX_STEPS, MISS_COST, LATENCY_WEIGHT, DELAY_PENALTY, REWARD_CLIP,
    K8S_NAMESPACE, TASK_IMAGE
)
from environment.workload import TaskGenerator, Task
from timing import PWCETEstimator
from metrics import MetricsTracker

try:
    from kubernetes import client, config as k8s_config
    from kubernetes.client.rest import ApiException
    K8S_LIB = True
except ImportError:
    K8S_LIB = False

NODE_F, TASK_F = 8, 6


class RealPodManager:
    """Handles lifecycle and poll operations for real inference task pods on Kubernetes."""
    
    def __init__(self):
        if not K8S_LIB:
            raise RuntimeError("kubernetes python SDK is required for K8s execution mode.")
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        
        self.core_v1 = client.CoreV1Api()
        try:
            self.core_v1.create_namespace(client.V1Namespace(metadata=client.V1ObjectMeta(name=K8S_NAMESPACE)))
        except ApiException as e:
            if e.status != 409:
                raise
                
        self.inflight: Dict[str, dict] = {}

    def submit_task_pod(self, task: Task, node_idx: Optional[int]) -> Tuple[bool, str]:
        """Creates a real task pod bound directly to a target node name."""
        node_name = NODE_PROFILES[node_idx]["name"] if node_idx is not None else None
        pod_name = f"{task.ttype}-{task.tid}-{int(time.time()*1000) % 10**7}"
        res = {"cpu": f"{int(task.cpu * 1000)}m", "memory": f"{task.mem_mb}Mi"}
        
        spec = {
            "containers": [{
                "name": "task",
                "image": TASK_IMAGE,
                "imagePullPolicy": "IfNotPresent",
                "env": [
                    {"name": "WORK_CS", "value": str(task.work)},
                    {"name": "CPU_LIMIT", "value": str(task.cpu)}
                ],
                "resources": {"requests": res, "limits": res},
            }],
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 0,
        }
        if node_name:
            spec["nodeName"] = node_name

        manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": pod_name,
                "namespace": K8S_NAMESPACE,
                "labels": {"app": "rt-bench", "ttype": task.ttype, "cls": task.cls}
            },
            "spec": spec
        }
        
        t_submit = time.time()
        try:
            self.core_v1.create_namespaced_pod(namespace=K8S_NAMESPACE, body=manifest)
            self.inflight[pod_name] = {"task": task, "node_idx": node_idx, "t_submit": t_submit}
            return True, pod_name
        except ApiException as e:
            print(f"  [!] Failed to launch pod {pod_name}: {e.status} {e.reason}")
            return False, ""

    def poll_completions(self) -> List[dict]:
        """Reads logs and completion statuses of active pods."""
        if not self.inflight:
            return []
        
        completions = []
        now = time.time()
        label_selector = "app=rt-bench"
        
        try:
            pods = self.core_v1.list_namespaced_pod(K8S_NAMESPACE, label_selector=label_selector).items
        except ApiException:
            return []

        for p in pods:
            info = self.inflight.get(p.metadata.name)
            if not info:
                continue
            
            phase = p.status.phase
            timed_out = (now - info["t_submit"]) > 3.0 * info["task"].deadline
            if phase not in ("Succeeded", "Failed") and not timed_out:
                continue

            rec = {
                "tid": info["task"].tid,
                "node_idx": info["node_idx"],
                "latency": now - info["t_submit"],
                "failed": phase != "Succeeded" or timed_out
            }

            if phase == "Succeeded":
                try:
                    txt = self.core_v1.read_namespaced_pod_log(p.metadata.name, K8S_NAMESPACE)
                    last_line = [l for l in txt.strip().split("\n") if l.strip()][-1]
                    log_data = json.loads(last_line)
                    rec["latency"] = log_data["t_end"] - info["t_submit"]
                except Exception:
                    pass

            completions.append(rec)
            self._delete_pod(p.metadata.name)
            self.inflight.pop(p.metadata.name, None)

        return completions

    def _delete_pod(self, pod_name: str):
        try:
            self.core_v1.delete_namespaced_pod(pod_name, K8S_NAMESPACE, grace_period_seconds=0)
        except ApiException:
            pass

    def cleanup_all(self):
        try:
            self.core_v1.delete_collection_namespaced_pod(K8S_NAMESPACE, label_selector="app=rt-bench")
        except Exception:
            pass
        self.inflight.clear()


class TaskOffloading(gym.Env):
    """
    Gymnasium-compliant environment for DRL-based (MATD3) real Kubernetes pod offloading 
    and deadline-aware resource scheduling.
    """
    metadata = {"render_modes": ["human"], "name": "k8s_matd3_rt_v1"}

    def __init__(self, pwcet_method: str = "empirical", seed: int = 42):
        super().__init__()
        self.N = NUM_NODES
        self.cores = np.array([n["cores"] for n in NODE_PROFILES])
        self.mem_cap = np.array([n["mem_mb"] for n in NODE_PROFILES], dtype=float)
        
        self.rng = np.random.default_rng(seed)
        self.gen = TaskGenerator(self.rng)
        self.est = PWCETEstimator(pwcet_method)
        self.metrics = MetricsTracker()
        self.pod_mgr = RealPodManager()

        self.obs_dim = NODE_F + TASK_F + self.N + 2
        self.state_dim = self.N * NODE_F + TASK_F + 2
        
        # Multi-agent action space (one bid per node agent in [-1, 1])
        self.action_space = spaces.Box(-1.0, 1.0, (self.N,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 2.0, (self.N, self.obs_dim), dtype=np.float32)

        self._init_episode_state()
        self.pwcet = np.zeros(self.N)

    def _init_episode_state(self):
        self.cpu_alloc = np.zeros(self.N)
        self.mem_alloc = np.zeros(self.N)
        self.running = np.zeros(self.N)
        self.miss_ema = np.zeros(self.N)
        self.inflight_tasks = {}
        self.pending = []
        self.current = None
        self.step_count = 0
        self.mask = np.zeros(self.N, bool)
        self.res_ok = np.zeros(self.N, bool)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.pod_mgr.cleanup_all()
        self.gen.reset(self.rng)
        self.est.reset()
        self.metrics.reset()
        self._init_episode_state()

        self.pending.append(self.gen.sample(time.time()))
        self.current = self.pending.pop(0)

        obs, state = self._build_observation(self.current)
        return obs, {"state": state, "mask": self.mask.copy(), "rate": self.gen.rate}

    def _build_observation(self, task: Task):
        now = time.time()
        slack = max(task.deadline_abs - now, 0.0)

        cpu_ok = (self.cpu_alloc + task.cpu) <= (CPU_CAP * self.cores + 1e-9)
        mem_ok = (self.mem_alloc + task.mem_mb) <= (MEM_CAP * self.mem_cap + 1e-9)
        self.res_ok = cpu_ok & mem_ok

        pwcet = np.array([self.est.estimate(i, task.ttype, task.work, task.cpu) for i in range(self.N)])
        timing_ok = pwcet <= slack
        mask = self.res_ok & timing_ok
        
        if not mask.any() and self.res_ok.any():
            mask = self.res_ok.copy()  # Fallback: best-effort placement
            
        self.mask = mask

        ratio = np.clip(pwcet / max(slack, 0.05), 0.0, 2.0) / 2.0
        cpu_u = self.cpu_alloc / self.cores
        
        node_features = np.stack([
            cpu_u,
            self.mem_alloc / self.mem_cap,
            np.minimum(self.running / MAX_RUNNING_NORM, 1.0),
            ratio,
            np.ones(self.N),  # Node availability flag
            self.cores / self.cores.max(),
            self.miss_ema,
            mask.astype(float)
        ], axis=1)

        task_features = np.array([
            float(task.cls == "hard"),
            task.cpu / MAX_CPU_REQ,
            task.mem_mb / MAX_MEM_REQ,
            min(slack / MAX_DEADLINE, 1.0),
            task.work / MAX_WORK,
            min(len(self.pending) / MAX_PENDING, 1.0)
        ])

        summary = np.array([cpu_u.mean(), cpu_u.std()])
        eye = np.eye(self.N)

        obs = np.concatenate([
            node_features,
            np.tile(task_features, (self.N, 1)),
            eye,
            np.tile(summary, (self.N, 1))
        ], axis=1)

        state = np.concatenate([node_features.ravel(), task_features, summary])
        self.pwcet = np.array([self.est.estimate(i, task.ttype, task.work, task.cpu) for i in range(self.N)])
        timing_ok = self.pwcet <= slack
        return obs.astype(np.float32), state.astype(np.float32)

    def step(self, action: Optional[np.ndarray]):
        task = self.current
        reward, delayed, placed = 0.0, False, -1
        self.step_count += 1

        # ---- Determine placement based on DRL agent bids and action masking
        if action is not None and self.mask.any():
            bids = np.where(self.mask, np.asarray(action, dtype=float).ravel(), -np.inf)
            target_node = int(np.argmax(bids))
            
            ok, _ = self.pod_mgr.submit_task_pod(task, target_node)
            if ok:
                placed = target_node

        if placed >= 0:
            self.inflight_tasks[task.tid] = (task, placed)
            self.cpu_alloc[placed] += task.cpu
            self.mem_alloc[placed] += task.mem_mb
            self.running[placed] += 1
        else:
            delayed = True
            reward -= DELAY_PENALTY
            self.pending.append(task)

        # ---- Poll for real pod completions on K8s cluster
        time.sleep(min(0.05, self.gen.next_interarrival()))
        completions = self.pod_mgr.poll_completions()
        reward += self._settle_completions(completions)

        # ---- Sample new incoming workload arrivals
        self.pending.append(self.gen.sample(time.time()))

        # ---- Drop expired/over-deadline tasks
        reward += self._drop_expired()

        truncated = self.step_count >= MAX_STEPS
        if truncated:
            self.pod_mgr.cleanup_all()
            for t in self.pending:
                reward -= MISS_COST[t.cls]

        reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
        self.metrics.add_step(reward, delayed)

        info = {"delayed": delayed, "step": self.step_count}
        if truncated:
            obs = np.zeros((self.N, self.obs_dim), dtype=np.float32)
            info.update(
                state=np.zeros(self.state_dim, dtype=np.float32),
                mask=np.zeros(self.N, dtype=bool),
                metrics=self.metrics.summary()
            )
            return obs, reward, False, True, info

        self.current = self.pending.pop(0)
        obs, state = self._build_observation(self.current)
        info.update(state=state, mask=self.mask.copy())

        return obs, reward, False, False, info

    def _settle_completions(self, completions: List[dict]) -> float:
        reward = 0.0
        for c in completions:
            tid = c["tid"]
            if tid not in self.inflight_tasks:
                continue
            
            task, node_idx = self.inflight_tasks.pop(tid)
            self.cpu_alloc[node_idx] = max(0.0, self.cpu_alloc[node_idx] - task.cpu)
            self.mem_alloc[node_idx] = max(0.0, self.mem_alloc[node_idx] - task.mem_mb)
            self.running[node_idx] = max(0, self.running[node_idx] - 1)

            missed = c["failed"] or (c["latency"] > task.deadline)
            if not c["failed"]:
                self.est.update(node_idx, task.ttype, c["latency"])

            self.miss_ema[node_idx] = 0.9 * self.miss_ema[node_idx] + 0.1 * float(missed)

            if missed:
                reward -= MISS_COST[task.cls]
            else:
                reward -= LATENCY_WEIGHT * (c["latency"] / task.deadline)

            self.metrics.add_completion(task.cls, c["latency"] if not c["failed"] else float("nan"), missed)

        return reward

    def _drop_expired(self) -> float:
        now = time.time()
        reward = 0.0
        remaining = []

        for t in self.pending:
            if t.deadline_abs < now:
                reward -= MISS_COST[t.cls]
                self.metrics.add_completion(t.cls, float("nan"), True, dropped=True)
            else:
                remaining.append(t)

        self.pending = remaining
        while len(self.pending) > MAX_PENDING:
            self.pending.sort(key=lambda x: x.deadline_abs)
            dropped_task = self.pending.pop(0)
            reward -= MISS_COST[dropped_task.cls]
            self.metrics.add_completion(dropped_task.cls, float("nan"), True, dropped=True)

        return reward

    def close(self):
        self.pod_mgr.cleanup_all()