"""Phase 2B settings and launch bounds, independent of the Phase 2A protocol."""
from pathlib import Path

from goalreach.curriculum import DEFAULTS

PILOT = 'p2b-pilot-balanced-R0-sgs-seed42-u8000'
VERIFY_BUDGETS = {2: 64, 20: 4096, 100: 4096, 200: 4096}
POOL_CONTENT = {
    'pools_v4_entry/train.npz': '64d0c6c391c2e3dfcb228d038e12d3d04c36936afe935717c229e694e79c13ad',
    'pools_v5_natural/evaluation.npz': '9f6ba65dc40050d8c48187631a0f5f499f347a8dc5acafdb880dbfe5d59fe716',
}
SGS = dict(history=100, target=.5, concentration=1., temperature=2., epsilon=1e-4)
SETTINGS = dict(phase='2B', reward='R0', num_envs=4096, updates=8000,
    rollout_steps=24, from_scratch=True, timeout_penalty=-50., ema_decay=.99,
    evaluation_every_updates=100, raw_evaluation_every_updates=500,
    checkpoint_every_updates=500, evaluation_pool_size=2016, training_pool_size=32768,
    success_criterion='hold_only', initial_action_std=.3, entropy_coef=.001,
    gamma=.99, curriculum=DEFAULTS, sgs=SGS)


def validate(config, verify, num_envs, label, device):
    for key, value in SETTINGS.items():
        if config.get(key) != value:
            raise ValueError('2B protocol mismatch: ' + key)
    for (role, (path, digest)) in zip(('training', 'evaluation'), POOL_CONTENT.items()):
        if config.get(role+'_pool') != path or config.get(role+'_pool_content_sha256') != digest:
            raise ValueError('2B pool binding mismatch: ' + role)
    seed = config.get('student_seed')
    if type(seed) is not int or seed not in (42, 43, 44, 45):
        raise ValueError('Invalid 2B student seed')
    if seed != config.get('environment_seed') or seed != config.get('sampler_seed'):
        raise ValueError('2B uses the student seed for the environment and sampler')
    if config.get('sampler') not in ('uniform', 'manual', 'sgs'):
        raise ValueError('Invalid 2B sampler')
    if not str(device).startswith('cuda:'):
        raise ValueError('2B training requires CUDA')
    if verify is not None:
        if config.get('launch_status') not in ('validation_only', 'yanglab_pilot_only'):
            raise ValueError('The selected config is pending the main pool decision')
        if verify not in VERIFY_BUDGETS or num_envs != VERIFY_BUDGETS[verify] or label == PILOT:
            raise ValueError('Invalid 2B verification budget, environment count, or output label')
        return
    if (label, seed, config['sampler'], config.get('launch_status'), num_envs, device) != (
            PILOT, 42, 'sgs', 'yanglab_pilot_only', 4096, 'cuda:0'):
        raise ValueError('The only permitted long run is the Yanglab seed 42 SGS pilot')


def verify_pilot_release(project, config):
    import hashlib
    import json

    release = json.loads((project/'delivery/p2b_release.json').read_text())
    if release.get('ready_for_rental_acceptance') is not True or release['pilot']['label'] != PILOT:
        raise ValueError('The frozen release must pass GPU acceptance before the pilot')
    if release['pilot']['protocol'] != config:
        raise ValueError('Pilot config differs from the frozen release')
    for name, expected in release['files_sha256'].items():
        if hashlib.sha256((project/Path(name)).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen release file changed: ' + name)
    for name, binding in release['pools'].items():
        if hashlib.sha256((project/Path(name)).read_bytes()).hexdigest() != binding['file_sha256']:
            raise ValueError('Frozen pool file changed: ' + name)
