# agents/matd3.py - MATD3 with one agent per node (CTDE)
#
#  * actor (decentralised): local obs of node i -> bid in [-1, 1] ("how much do I want this task")
#  * critic (centralised, twin): global state + all N (masked) bids -> Q
#  * placement = argmax over feasible bids, so the discrete choice never needs a softmax relaxation
#  * infeasible nodes (resource / pWCET mask) have their bid fixed to -1 in actor loss, TD target and replay
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class ActorBank(nn.Module):
    """share=True: one network for all agents (the one-hot id + capacity features differentiate nodes)."""
    def __init__(self, n_agents, obs_dim, hidden, share=True):
        super().__init__()
        self.n, self.share = n_agents, share
        self.nets = nn.ModuleList([mlp(obs_dim, hidden, 1) for _ in range(1 if share else n_agents)])

    def forward(self, obs):                                   # obs: (B, N, d) -> (B, N)
        if self.share:
            return torch.tanh(self.nets[0](obs)).squeeze(-1)
        return torch.stack([torch.tanh(self.nets[i](obs[:, i])).squeeze(-1) for i in range(self.n)], dim=1)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, n_agents, hidden):
        super().__init__()
        self.q1, self.q2 = mlp(state_dim + n_agents, hidden, 1), mlp(state_dim + n_agents, hidden, 1)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, s, a):
        return self.q1(torch.cat([s, a], dim=-1))


class MATD3:
    def __init__(self, n_agents, obs_dim, state_dim, cfg: dict, device="cpu"):
        self.n, self.cfg, self.dev = n_agents, cfg, torch.device(device)
        torch.set_num_threads(2)                              # keep the laptop responsive
        self.actor = ActorBank(n_agents, obs_dim, cfg["hidden"], cfg["share_params"]).to(self.dev)
        self.critic = TwinCritic(state_dim, n_agents, cfg["hidden"]).to(self.dev)
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
        self.a_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.c_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.updates = 0

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_std: float = 0.0, rng=None) -> np.ndarray:
        a = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=self.dev).unsqueeze(0))[0].cpu().numpy()
        if noise_std > 0:
            rng = rng or np.random.default_rng()
            a = a + rng.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------ learning
    def update(self, b: dict) -> dict:
        c, T = self.cfg, lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.dev)
        obs, state, act, mask = T(b["obs"]), T(b["state"]), T(b["act"]), T(b["mask"])
        rew, nobs, nstate, nmask, done = T(b["rew"]), T(b["nobs"]), T(b["nstate"]), T(b["nmask"]), T(b["done"])
        self.updates += 1

        with torch.no_grad():                                  # target policy smoothing + clipped double Q
            na = self.actor_t(nobs)
            noise = (torch.randn_like(na) * c["policy_noise"]).clamp(-c["noise_clip"], c["noise_clip"])
            na = torch.where(nmask > 0.5, (na + noise).clamp(-1, 1), torch.full_like(na, -1.0))
            q1t, q2t = self.critic_t(nstate, na)
            y = rew + c["gamma"] * (1.0 - done) * torch.min(q1t, q2t)

        q1, q2 = self.critic(state, act)
        c_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.c_opt.zero_grad(); c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.c_opt.step()

        a_loss = torch.zeros(())
        if self.updates % c["policy_delay"] == 0:              # delayed actor + target updates
            a = self.actor(obs)
            a = torch.where(mask > 0.5, a, torch.full_like(a, -1.0))
            a_loss = -self.critic.q1_only(state, a).mean()
            self.a_opt.zero_grad(); a_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.a_opt.step()
            self._soft_update(self.actor, self.actor_t)
            self._soft_update(self.critic, self.critic_t)
        return {"critic_loss": float(c_loss), "actor_loss": float(a_loss)}

    def _soft_update(self, net, tgt):
        tau = self.cfg["tau"]
        for p, tp in zip(net.parameters(), tgt.parameters()):
            tp.data.mul_(1 - tau).add_(tau * p.data)

    # ------------------------------------------------------------------ io
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.dev)
        self.actor.load_state_dict(d["actor"]); self.critic.load_state_dict(d["critic"])
        self.actor_t, self.critic_t = copy.deepcopy(self.actor), copy.deepcopy(self.critic)
