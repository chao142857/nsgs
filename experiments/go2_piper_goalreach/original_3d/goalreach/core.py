"""Simulator-independent task semantics, also used by the live environment."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .vendor_sampler import TaskSampler

REGIONS = ('near', 'inplace', 'step', 'walk')
PARAMETERS = dict(history=100, target=.5, concentration=1., temperature=2., epsilon=1e-4)
REASONS = ('active', 'success', 'timeout', 'base_contact', 'tip', 'nonfoot_contact')


def failure_reward(reason, timeout_penalty=-100.):
    """Actual terminal reward; physical failure always costs 100."""
    if timeout_penalty not in (-100., -50.):
        raise ValueError('Prepared timeout penalties are -100 and -50')
    return torch.where(reason == 2, float(timeout_penalty),
                       torch.where(reason >= 3, -100., 0.))


def inverse_rotate(q, v):
    """Quaternion wxyz, world vector -> full body frame (including pitch/roll)."""
    xyz = -q[..., 1:]
    t = 2 * torch.cross(xyz, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(xyz, t, dim=-1)


def body_vectors(q, base, ee, target):
    # One metre units near the goal; radial clamp preserves direction at distance.
    def scaled(v):
        return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp(min=1.)
    return scaled(inverse_rotate(q, target-base)), scaled(inverse_rotate(q, target-ee))


class EpisodeState:
    def __init__(self, n, device='cpu', dt=.02, success_criterion='hold_and_speed'):
        if success_criterion not in ('hold_and_speed', 'hold_only'):
            raise ValueError(f'Unknown success criterion: {success_criterion}')
        self.hold_only = success_criterion == 'hold_only'
        self.n, self.device, self.dt = n, device, dt
        assert abs(dt-.02) < 1e-9, '25 control steps must equal 0.5 seconds'
        self.target = torch.zeros(n, 3, device=device)
        self.task = torch.full((n,), -1, dtype=torch.long, device=device)
        self.steps = torch.zeros(n, dtype=torch.long, device=device)
        self.hold = torch.zeros_like(self.steps)
        self.reason = torch.zeros_like(self.steps)
        self.generation = torch.zeros_like(self.steps)
        self.distance = torch.full((n,), float('inf'), device=device)
        self.minimum = self.distance.clone()
        self.previous = self.distance.clone()
        self.progress = torch.zeros(n, device=device)
        if self.hold_only:
            self.hold_speed_max = torch.zeros_like(self.distance)
            self.hold_speed_sum = torch.zeros_like(self.distance)

    def reset(self, ids, tasks, targets):
        self.task[ids] = tasks
        self.target[ids] = targets
        self.steps[ids] = self.hold[ids] = self.reason[ids] = 0
        self.distance[ids] = self.minimum[ids] = self.previous[ids] = float('inf')
        self.progress[ids] = 0
        if self.hold_only:
            self.hold_speed_max[ids] = self.hold_speed_sum[ids] = 0
        self.generation[ids] += 1

    def advance(self, ee, velocity, base_contact, tipped, nonfoot):
        active = (self.task >= 0) & (self.reason == 0)
        d = (self.target-ee).norm(dim=-1)
        speed = velocity.norm(dim=-1)
        self.steps += active.long()
        self.distance = torch.where(active, d, self.distance)
        self.minimum = torch.where(active, torch.minimum(self.minimum, d), self.minimum)
        self.progress = torch.where(active & torch.isfinite(self.previous),
                                    (self.previous-d).clamp(-.02, .02), 0.)
        self.previous = torch.where(active, d, self.previous)
        good = (d <= .05) & active
        if self.hold_only:
            self.hold_speed_max = torch.where(active,
                torch.where(good, torch.maximum(self.hold_speed_max, speed), 0.), self.hold_speed_max)
            self.hold_speed_sum = torch.where(active,
                torch.where(good, self.hold_speed_sum + speed, 0.), self.hold_speed_sum)
        else:
            good = good & (speed < .25)
        self.hold = torch.where(good, self.hold+1, torch.zeros_like(self.hold))
        failure = torch.where(base_contact, 3, torch.where(tipped, 4, torch.where(nonfoot, 5, 0)))
        success = (self.hold >= 25) & (failure == 0) & (self.steps <= 500)
        reason = torch.where(failure != 0, failure,
                             torch.where(success, 1, torch.where(self.steps >= 500, 2, 0)))
        self.reason = torch.where(active, reason, self.reason)
        return self.reason != 0

    def rows(self, ids):
        columns = torch.stack([self.task, self.generation, self.steps, self.reason,
                               self.distance, self.minimum], -1)[ids].cpu().tolist()
        result = []
        if self.hold_only:
            speeds = torch.stack([self.hold_speed_max, self.hold_speed_sum / 25], -1)[ids].cpu().tolist()
        for i, (task, gen, steps, reason, distance, minimum) in enumerate(columns):
            if int(reason) == 0:
                raise ValueError('Cannot report an unfinished episode')
            result.append(dict(task_id=int(task), generation=int(gen), success=int(reason)==1,
                failure_reason=None if int(reason)==1 else REASONS[int(reason)],
                episode_steps=int(steps), terminal_distance_m=distance, minimum_distance_m=minimum))
            if self.hold_only and int(reason) == 1:
                result[-1]['hold_speed_max_m_s'], result[-1]['hold_speed_mean_m_s'] = speeds[i]
        return result


class Sampler(TaskSampler):
    """Exact inherited SGS history/kernel, vectorized probability computation."""
    def __init__(self, regions, seed, mode='uniform', region=None):
        if mode not in ('uniform', 'sgs', 'external', 'evaluation'):
            raise ValueError(mode)
        super().__init__(len(regions), seed, mode='sgs' if mode=='sgs' else 'uniform', **PARAMETERS)
        self.interface_mode = mode
        self.regions = np.asarray(regions)
        self.allowed = np.ones(self.n, bool) if region is None else self.regions == REGIONS.index(region)
        if not self.allowed.any():
            raise ValueError('Empty region')
        self.external = None
        self.cached_rates = np.full(self.n, .5)

    def set_probabilities(self, values):
        p = np.asarray(values, dtype=np.float64)
        if p.shape != (self.n,) or not np.isfinite(p).all() or (p < 0).any() or not np.isfinite(p.sum()) or p.sum() <= 0:
            raise ValueError('Expected finite nonnegative pool-sized probability vector with positive mass')
        if (p*self.allowed).sum() <= 0:
            raise ValueError('No mass in selected region')
        self.external = p/p.sum()

    def probabilities(self):
        if self.interface_mode == 'sgs':
            p, e = self.cached_rates, self.epsilon
            w = (p+e)**(self.concentration*self.target) * (1-p+e)**(self.concentration*(1-self.target))
            logits = np.log(np.maximum(w,e)+e)/self.temperature
            p = np.exp(logits-logits.max())
        elif self.interface_mode == 'external':
            if self.external is None:
                raise ValueError('External probabilities have not been set')
            p = self.external.copy()
        else:
            p = np.ones(self.n)
        p = p*self.allowed
        return (p/p.sum()).tolist()

    def sample(self, count):
        if self.interface_mode == 'evaluation':
            candidates = np.flatnonzero(self.allowed)
            ids = candidates[(np.arange(count)+self.eval_cursor) % len(candidates)].tolist()
            self.eval_cursor += count
            for i in ids:
                self.sample_counts[i] += 1
            return ids
        return super().sample(count)

    def record(self, ids, successes):
        if self.interface_mode == 'evaluation':
            return
        super().record(ids, successes)
        for i in set(ids):
            self.cached_rates[i] = sum(self.outcomes[i])/len(self.outcomes[i])

    def state_dict(self):
        s = super().state_dict()
        s.update(interface_mode=self.interface_mode, allowed=self.allowed.tolist(),
                 external=None if self.external is None else self.external.tolist())
        return s

    def load_state_dict(self, s):
        if s['interface_mode'] != self.interface_mode or s['allowed'] != self.allowed.tolist():
            raise ValueError('Sampler interface/region differs')
        super().load_state_dict(s)
        self.cached_rates = np.array(super().rates())
        if s['external'] is not None:
            self.set_probabilities(s['external'])


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def append_rows(path, rows):
    with Path(path).open('a') as f:
        for row in rows:
            f.write(json.dumps(row, allow_nan=False)+'\n')
