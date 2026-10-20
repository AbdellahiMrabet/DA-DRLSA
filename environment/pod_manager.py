# environment/pod_manager.py - launches finite "inference task" pods and measures real latency
#
# Adapted from the nginx/TTL pod manager: pods are now finite tasks (they exit when the
# work is done). The end-to-end response time is measured from just before the create
# call (host clock) to the container's own end timestamp (read from its log), so it
# contains API latency + scheduling/binding + image start (cold start) + execution.
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from config import K8S_NAMESPACE, TASK_IMAGE, NODE_PROFILES

try:
    from kubernetes import client, config as k8s_config
    from kubernetes.client.rest import ApiException
    K8S_LIB = True
except Exception:  # sim-only machines do not need the kubernetes package
    K8S_LIB = False

LABEL = "app=rt-bench"


class InferenceTaskManager:
    def __init__(self):
        self.core_v1 = None
        self.inflight: Dict[str, dict] = {}      # pod_name -> {task, node_idx, t_submit, api_ms}
        self.api_latencies: List[float] = []
        self._ready_cache: Tuple[float, np.ndarray] = (0.0, np.ones(len(NODE_PROFILES), bool))
        self._init_client()

    # ------------------------------------------------------------------ setup
    def _init_client(self):
        if not K8S_LIB:
            raise RuntimeError("pip install kubernetes  (needed for the k8s backend)")
        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        self.core_v1 = client.CoreV1Api()
        try:
            self.core_v1.create_namespace(
                client.V1Namespace(metadata=client.V1ObjectMeta(name=K8S_NAMESPACE)))
        except ApiException as e:
            if e.status != 409:      # 409 = already exists
                raise

    # ------------------------------------------------------------------ submit
    def submit(self, task, node_name: Optional[str], node_idx: int = -1) -> Tuple[bool, str, float]:
        """Create the pod. node_name=None -> let the default kube-scheduler decide."""
        pod_name = f"{task.ttype}-{task.tid}-{int(time.time()*1000) % 10**7}-{np.random.randint(100, 999)}"
        res = {"cpu": f"{int(task.cpu*1000)}m", "memory": f"{task.mem_mb}Mi"}
        spec = {
            "containers": [{
                "name": "task", "image": TASK_IMAGE, "imagePullPolicy": "IfNotPresent",
                "env": [{"name": "WORK_CS", "value": str(task.work)},
                        {"name": "CPU_LIMIT", "value": str(task.cpu)}],
                "resources": {"requests": res, "limits": res},
            }],
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 0,
        }
        if node_name is not None:
            spec["nodeName"] = node_name                 # direct binding (bypasses kube-scheduler)
        manifest = {"apiVersion": "v1", "kind": "Pod",
                    "metadata": {"name": pod_name, "namespace": K8S_NAMESPACE,
                                 "labels": {"app": "rt-bench", "ttype": task.ttype, "cls": task.cls}},
                    "spec": spec}
        t_submit = time.time()
        t0 = time.perf_counter()
        try:
            self.core_v1.create_namespaced_pod(namespace=K8S_NAMESPACE, body=manifest)
        except ApiException as e:
            print(f"  [!] create pod failed: {e.status} {e.reason}")
            return False, "", 0.0
        api_ms = (time.perf_counter() - t0) * 1000
        self.api_latencies.append(api_ms)
        self.inflight[pod_name] = dict(task=task, node_idx=node_idx, t_submit=t_submit, api_ms=api_ms)
        return True, pod_name, api_ms

    def resolve_binding(self, pod_name: str, timeout: float = 3.0) -> int:
        """Node index the default scheduler picked (-1 if not bound in time)."""
        names = [n["name"] for n in NODE_PROFILES]
        t_end = time.time() + timeout
        while time.time() < t_end:
            try:
                p = self.core_v1.read_namespaced_pod(pod_name, K8S_NAMESPACE)
                if p.spec.node_name:
                    idx = names.index(p.spec.node_name) if p.spec.node_name in names else -1
                    self.inflight[pod_name]["node_idx"] = idx
                    return idx
            except ApiException:
                pass
            time.sleep(0.05)
        return -1

    # ------------------------------------------------------------------ poll
    def poll(self) -> List[dict]:
        """One API call. Returns finished tasks as dicts and deletes their pods."""
        if not self.inflight:
            return []
        done = []
        now = time.time()
        pods = self.core_v1.list_namespaced_pod(K8S_NAMESPACE, label_selector=LABEL).items
        names = [n["name"] for n in NODE_PROFILES]
        for p in pods:
            info = self.inflight.get(p.metadata.name)
            if info is None:
                continue
            phase = p.status.phase
            timed_out = (now - info["t_submit"]) > 3.0 * info["task"].deadline
            if phase not in ("Succeeded", "Failed") and not timed_out:
                continue
            nn = p.spec.node_name
            node_idx = names.index(nn) if nn in names else info["node_idx"]
            rec = dict(pod=p.metadata.name, task=info["task"], node_idx=node_idx, failed=False,
                       latency=now - info["t_submit"], startup=None, exec_s=None)
            if phase == "Succeeded":
                m = self._read_log(p.metadata.name)
                if m:
                    rec.update(latency=m["t_end"] - info["t_submit"],
                               startup=m["t_start"] - info["t_submit"], exec_s=m["exec_s"])
            else:
                rec["failed"] = True          # Failed or timed out
            done.append(rec)
            self._delete(p.metadata.name)
            self.inflight.pop(p.metadata.name, None)
        return done

    def _read_log(self, name: str) -> Optional[dict]:
        try:
            txt = self.core_v1.read_namespaced_pod_log(name, K8S_NAMESPACE)
            line = [l for l in txt.strip().split("\n") if l.strip()][-1]
            return json.loads(line)
        except Exception:
            return None

    def _delete(self, name: str):
        try:
            self.core_v1.delete_namespaced_pod(name, K8S_NAMESPACE, grace_period_seconds=0)
        except ApiException:
            pass

    # ------------------------------------------------------------------ cluster state
    def node_ready(self, max_age: float = 5.0) -> np.ndarray:
        t, cached = self._ready_cache
        if time.time() - t < max_age:
            return cached
        ready = np.zeros(len(NODE_PROFILES), dtype=bool)
        try:
            by_name = {n.metadata.name: n for n in self.core_v1.list_node().items}
            for i, prof in enumerate(NODE_PROFILES):
                n = by_name.get(prof["name"])
                if n is None:
                    continue
                for c in n.status.conditions or []:
                    if c.type == "Ready":
                        ready[i] = (c.status == "True")
        except Exception:
            ready[:] = True
        self._ready_cache = (time.time(), ready)
        return ready

    def get_avg_api_latency(self) -> float:
        return float(np.mean(self.api_latencies)) if self.api_latencies else 0.0

    def cleanup_all(self):
        try:
            self.core_v1.delete_collection_namespaced_pod(K8S_NAMESPACE, label_selector=LABEL)
        except Exception as e:
            print(f"  [!] cleanup warning: {e}")
        self.inflight.clear()
