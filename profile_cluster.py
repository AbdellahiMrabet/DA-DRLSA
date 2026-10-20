# profile_cluster.py - measure startup/exec distributions on the REAL cluster -> calibration.json
# Run on an idle cluster. Tasks are submitted one at a time per node so nothing interferes.
import argparse, json, time
import numpy as np
from config import NODE_PROFILES, TASK_TYPES, CALIBRATION_FILE
from environment.backends import K8sBackend
from environment.workload import Task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30, help="tasks per (node, type)")
    a = ap.parse_args()
    be = K8sBackend(); be.reset()
    startup = {i: [] for i in range(len(NODE_PROFILES))}
    ratio = []
    tid = 0
    for i, prof in enumerate(NODE_PROFILES):
        for name, t in TASK_TYPES.items():
            if t["cpu"] > prof["cores"] * 0.95:
                continue
            for _ in range(a.reps):
                tid += 1
                task = Task(tid, name, t["cls"], t["cpu"], t["mem_mb"], t["work"], t["deadline"], 0.0)
                be.place(task, i)
                done = []
                while not done:
                    done = be.advance(0.3)
                c = done[0]
                if c.startup and c.exec_s and not c.failed:
                    startup[i].append(c.startup)
                    ratio.append(c.exec_s / (t["work"] / min(t["cpu"], prof["cores"])))
            print(f"node {i} {name}: ok")
    cal = {
        "startup_median_s": [float(np.median(v)) if v else 1.0 for v in startup.values()],
        "startup_sigma": float(np.mean([np.std(np.log(v)) for v in startup.values() if len(v) > 3])),
        "exec_sigma": float(np.std(np.log(ratio))) if ratio else 0.1,
    }
    CALIBRATION_FILE.write_text(json.dumps(cal, indent=2))
    print("wrote", CALIBRATION_FILE, cal)
    be.reset()


if __name__ == "__main__":
    main()
