"""Frozen phase 2A specifications; reward is exclusively a required launch argument."""
from math import isfinite

from goalreach.curriculum import DEFAULTS

def validate(d):
    expected=dict(phase='2A',num_envs=4096,rollout_steps=24,student_seed=42,
        from_scratch=True,timeout_penalty=-50.,ema_decay=.99,evaluation_every_updates=100,
        raw_evaluation_every_updates=500,checkpoint_every_updates=500,evaluation_pool_size=2016,
        curriculum=DEFAULTS)
    for k,v in expected.items():
        if d.get(k)!=v:raise ValueError('2A protocol mismatch: '+k)
    if d.get('updates') not in (5000, 8000):
        raise ValueError('2A protocol mismatch: updates')
    pools = {'A1': 'pools_v4_entry/train.npz', 'A2': 'pools_v4_entry/train.npz',
             'A3': 'pools_v5_natural/train.npz', 'A4': 'pools_v5_natural/train.npz'}
    if d['updates'] == 8000 and (d.get('arm') not in pools or d.get('training_pool') != pools[d['arm']]):
        raise ValueError('Incorrect arm or pool for the 8,000-update budget')
    if d.get('reward') is not None:raise ValueError('Reward must be supplied explicitly at launch')
    if d['sampler'] not in ('uniform','manual'):raise ValueError('Unsupported 2A sampler')
    if d['initial_action_std']<=0 or not 0<d['gamma']<=1:raise ValueError('Invalid PPO parameters')

    entropy_coef = d.get('entropy_coef', .005)
    if not isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError('PPO entropy coefficient must be finite and nonnegative')

    if d.get('success_criterion', 'hold_and_speed') not in ('hold_and_speed', 'hold_only'):
        raise ValueError('Unknown success criterion')
