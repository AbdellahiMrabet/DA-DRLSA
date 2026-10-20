# evaluate.py - compare policies on identical task streams (same seeds), sim or real k8s
import argparse, json, time
import numpy as np
from environment.env import TaskOffloading
from agents.baselines import BASELINES


class MATD3Policy:
    name = "matd3"
    def __init__(self, env, ckpt):
        from config import MATD3 as C
        from agents.matd3 import MATD3
        self.agent = MATD3(env.N, env.obs_dim, env.state_dim, C)
        self.agent.load(ckpt)
    def reset(self): pass
    def act(self, env, obs): return self.agent.act(obs, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="sim", choices=["sim", "k8s"], help="Execution backend")
    ap.add_argument("--policies", nargs="+", default=["round_robin", "least_loaded", "fastest_feasible", "matd3"])
    ap.add_argument("--ckpt", default="ckpt/matd3_best.pt")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--pwcet", default="empirical", choices=["empirical", "evt"])
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()

    env = TaskOffloading(pwcet_method=a.pwcet, seed=a.seed)
    results = {}
    for name in a.policies:
        pol = MATD3Policy(env, a.ckpt) if name == "matd3" else BASELINES[name]()
        per_ep = []
        for ep in range(a.episodes):
            pol.reset()
            obs, info = env.reset(seed=a.seed + ep)
            while True:
                t = time.perf_counter()
                act = pol.act(env, obs)
                env.metrics.add_decision_time((time.perf_counter() - t) * 1000)
                obs, r, term, trunc, info = env.step(act)
                if trunc:
                    per_ep.append(info["metrics"]); break
        def agg(f):
            v = np.array([f(m) for m in per_ep], dtype=float)
            return [float(np.nanmean(v)), float(np.nanstd(v))]
        results[name] = {
            "miss_all": agg(lambda m: m["all"]["miss_ratio"]),
            "miss_hard": agg(lambda m: m["hard"]["miss_ratio"]),
            "miss_soft": agg(lambda m: m["soft"]["miss_ratio"]),
            "p99_hard_s": agg(lambda m: m["hard"].get("p99", np.nan)),
            "p99_all_s": agg(lambda m: m["all"].get("p99", np.nan)),
            "delays": agg(lambda m: m["delays"]),
            "decision_p99_ms": agg(lambda m: m["decision_ms"]["p99"]),
        }
        r_ = results[name]
        print(f"{name:17s} miss(all/hard/soft)={r_['miss_all'][0]:.3f}/{r_['miss_hard'][0]:.3f}/{r_['miss_soft'][0]:.3f}"
              f"  p99_hard={r_['p99_hard_s'][0]:.2f}s  decision_p99={r_['decision_p99_ms'][0]:.3f}ms")
    json.dump(results, open(a.out, "w"), indent=2)
    env.close()


if __name__ == "__main__":
    main()