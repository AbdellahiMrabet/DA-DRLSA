# train.py - train MATD3 using the real Kubernetes pod environment
import argparse, json, os, time
import numpy as np
from config import MATD3 as C
from environment.env import TaskOffloading
from agents.matd3 import MATD3
from agents.replay import ReplayBuffer


def run_eval(agent, env, seeds):
    out = []
    for s in seeds:
        obs, info = env.reset(seed=s)
        while True:
            obs, r, term, trunc, info = env.step(agent.act(obs, 0.0))
            if trunc:
                out.append(info["metrics"]); break
    miss = np.mean([m["all"]["miss_ratio"] for m in out])
    hard = np.mean([m["hard"]["miss_ratio"] for m in out])
    return float(miss), float(hard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--load", default=None, help="checkpoint to fine-tune from")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    env = TaskOffloading(seed=a.seed)
    eval_env = TaskOffloading(seed=a.seed + 1)
    agent = MATD3(env.N, env.obs_dim, env.state_dim, C)
    warm = C["warmup_steps"]
    if a.load:
        agent.load(a.load); warm = 0
    buf = ReplayBuffer(C["buffer_size"], env.N, env.obs_dim, env.state_dim)

    total, best, log = 0, float("inf"), []
    t0 = time.time()
    for ep in range(a.episodes):
        frac = ep / max(a.episodes - 1, 1)
        noise = C["expl_noise_start"] + (C["expl_noise_end"] - C["expl_noise_start"]) * frac
        obs, info = env.reset(seed=a.seed * 100_000 + ep)
        state, mask = info["state"], info["mask"]
        ep_r, losses = 0.0, []
        while True:
            bids = rng.uniform(-1, 1, env.N).astype(np.float32) if total < warm else agent.act(obs, noise, rng)
            nobs, r, term, trunc, info = env.step(bids)
            nstate, nmask = info["state"], info["mask"]
            buf.add(obs, state, np.where(mask, bids, -1.0), mask.astype(np.float32), r,
                    nobs, nstate, nmask.astype(np.float32), float(term or trunc))
            obs, state, mask = nobs, nstate, nmask
            ep_r += r; total += 1
            if total >= warm and len(buf) >= C["batch_size"]:
                losses.append(agent.update(buf.sample(C["batch_size"], rng)))
            if trunc:
                break
        m = info["metrics"]
        row = dict(ep=ep, reward=ep_r, miss=m["all"]["miss_ratio"], hard=m["hard"]["miss_ratio"],
                   soft=m["soft"]["miss_ratio"], delays=m["delays"], dropped=m["dropped"],
                   closs=float(np.mean([l["critic_loss"] for l in losses])) if losses else None,
                   rate=info.get("rate"), t=round(time.time() - t0, 1))
        log.append(row)
        print(json.dumps(row))
        if (ep + 1) % a.eval_every == 0:
            miss, hard = run_eval(agent, eval_env, seeds=[900_000 + i for i in range(5)])
            print(f"[eval] ep {ep+1}: miss={miss:.3f} hard_miss={hard:.3f}")
            if miss < best:
                best = miss; agent.save(os.path.join(a.out, "matd3_best.pt"))
    agent.save(os.path.join(a.out, "matd3_last.pt"))
    json.dump(log, open(os.path.join(a.out, "train_log.json"), "w"))
    env.close()


if __name__ == "__main__":
    main()