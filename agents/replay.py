# agents/replay.py
import numpy as np


class ReplayBuffer:
    def __init__(self, size, n_agents, obs_dim, state_dim):
        self.size, self.ptr, self.full = size, 0, False
        z = lambda *s: np.zeros(s, dtype=np.float32)
        self.obs, self.nobs = z(size, n_agents, obs_dim), z(size, n_agents, obs_dim)
        self.state, self.nstate = z(size, state_dim), z(size, state_dim)
        self.act, self.mask, self.nmask = z(size, n_agents), z(size, n_agents), z(size, n_agents)
        self.rew, self.done = z(size, 1), z(size, 1)

    def add(self, obs, state, act, mask, rew, nobs, nstate, nmask, done):
        i = self.ptr
        self.obs[i], self.state[i], self.act[i], self.mask[i] = obs, state, act, mask
        self.rew[i], self.nobs[i], self.nstate[i], self.nmask[i], self.done[i] = rew, nobs, nstate, nmask, done
        self.ptr = (self.ptr + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, batch, rng):
        idx = rng.integers(0, len(self), size=batch)
        return {k: getattr(self, k)[idx] for k in
                ("obs", "state", "act", "mask", "rew", "nobs", "nstate", "nmask", "done")}
