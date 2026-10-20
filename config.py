# config.py - central configuration (edit this first)
import json
from pathlib import Path

ROOT = Path(__file__).parent

# ---------------------------------------------------------------- Kubernetes
K8S_NAMESPACE = "rt-bench"
TASK_IMAGE = "rt-task:latest"          # built from task_image/, imported with `k3d image import`
CALIBRATION_FILE = ROOT / "calibration.json"

# Emulated heterogeneous fog nodes. `name` must match the k3d agent node names and
# `cores` must match the docker --cpus limit applied in setup_cluster.sh.
NODE_PROFILES = [
    {"name": "k3d-rt-agent-0", "cores": 0.75, "mem_mb": 512},
    {"name": "k3d-rt-agent-1", "cores": 1.25, "mem_mb": 768},
    {"name": "k3d-rt-agent-2", "cores": 1.75, "mem_mb": 768},
]
NUM_NODES = len(NODE_PROFILES)

# Admission cap: never place more than CPU_CAP * cores of cpu requests on a node
CPU_CAP = 1.0
MEM_CAP = 0.90
MAX_RUNNING_NORM = 10           # normalisation constant for #running tasks

# ---------------------------------------------------------------- Workload
# work = core-seconds of CPU the task burns (compute-bound inference stand-in).
# Ideal execution time = work / cpu. Startup (cold start) is added on top.
TASK_TYPES = {
    "kws":     dict(cpu=0.25, mem_mb=64,  work=0.25, cls="hard", deadline=8.0,  prob=0.30),
    "anomaly": dict(cpu=0.50, mem_mb=96,  work=0.50, cls="hard", deadline=9.0,  prob=0.25),
    "vision":  dict(cpu=0.75, mem_mb=128, work=0.90, cls="soft", deadline=15.0, prob=0.30),
    "batch":   dict(cpu=0.50, mem_mb=128, work=1.50, cls="soft", deadline=20.0, prob=0.15),
}
MAX_CPU_REQ = max(t["cpu"] for t in TASK_TYPES.values())
MAX_MEM_REQ = max(t["mem_mb"] for t in TASK_TYPES.values())
MAX_WORK = max(t["work"] for t in TASK_TYPES.values())
MAX_DEADLINE = max(t["deadline"] for t in TASK_TYPES.values())

ARRIVAL_RATE_RANGE = (0.2, 0.8)  # tasks/s, sampled per episode (cluster saturates ~1.8)
BURST_PROB = 0.10                # probability that an inter-arrival uses a burst
BURST_FACTOR = 2.0
MAX_PENDING = 30                 # tasks waiting for a decision; beyond this -> dropped

# ---------------------------------------------------------------- Timing (pWCET)
PWCET_QUANTILE = 0.99
PWCET_WINDOW = 200
PWCET_MIN_SAMPLES = 8
PWCET_PRIOR_MARGIN = 1.3
STARTUP_PRIOR_S = 2.0            # pessimistic cold-start prior used before data exists

# ---------------------------------------------------------------- Reward
MISS_COST = {"hard": 5.0, "soft": 2.0}
LATENCY_WEIGHT = 0.3             # x (latency / deadline) for met deadlines
DELAY_PENALTY = 0.05
REWARD_CLIP = 10.0

# ---------------------------------------------------------------- Episode
MAX_STEPS = 200                  # scheduling decisions per episode
DRAIN_TIMEOUT_S = 40.0           # wait for in-flight tasks at episode end (k8s)

# ---------------------------------------------------------------- Simulator defaults
# Overridden by calibration.json (written by profile_cluster.py).
SIM_PARAMS = {
    "startup_median_s": [1.4, 1.1, 0.9],   # per node
    "startup_sigma": 0.35,
    "startup_load_factor": 0.5,            # startup *= 1 + f * utilisation
    "exec_sigma": 0.10,
    "interference_kappa": 0.5,            # speed /= 1 + kappa * util^2
    "stall_prob": 0.03,
    "stall_range": [1.5, 3.0],
    "dt": 0.05,
}
if CALIBRATION_FILE.exists():
    SIM_PARAMS.update(json.loads(CALIBRATION_FILE.read_text()))

# ---------------------------------------------------------------- MATD3
MATD3 = dict(
    hidden=128, actor_lr=3e-4, critic_lr=1e-3, gamma=0.95, tau=0.005,
    policy_noise=0.2, noise_clip=0.5, policy_delay=2,
    batch_size=128, buffer_size=100_000, warmup_steps=500,
    expl_noise_start=0.3, expl_noise_end=0.1, share_params=True,
)
