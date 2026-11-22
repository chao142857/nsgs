"""Admission for the frozen formal Phase 2B schedule."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from tools.p2b_protocol import SETTINGS, SGS

BASE_COMMIT = 'e1b5ceff7891dc584a67909cfbe12a41ab15090a'
DECISION = 'delivery/p2b_main_pool_decision.json'
RELEASE = 'delivery/p2b_formal_release.json'
ACCEPTANCE = 'runtime/p2b_formal_entry_acceptance.json'
ARMS = ('uniform', 'sgs', 'manual')
SEEDS = {'yanglab': (43,), 'rental-4060ti': (44, 45)}
GPU = {'yanglab': 'NVIDIA GeForce RTX 3060', 'rental-4060ti': 'NVIDIA GeForce RTX 4060 Ti'}
TRAINING = {
    'balanced': dict(path='pools_v4_entry/train.npz', size=32768,
        content_sha256='64d0c6c391c2e3dfcb228d038e12d3d04c36936afe935717c229e694e79c13ad',
        file_sha256='fccd34a5ee16cfd892db32da548c9593e7cca6b26260347872ceb345ac2e5a93'),
    'natural': dict(path='pools_v5_natural/train.npz', size=32768,
        content_sha256='434b9b149d886b26c802548306fb80653fae78231aa092f271dbc0eb335af176',
        file_sha256='b886de8ba3e0735bb6ee7c2645368e14c8f498086c400e9eaf5eb1e1920aaf23'),
}
EVALUATION = dict(path='pools_v5_natural/evaluation.npz', size=2016,
    content_sha256='9f6ba65dc40050d8c48187631a0f5f499f347a8dc5acafdb880dbfe5d59fe716',
    file_sha256='d50591e1af543a1f3ff1a2630d624e457a02bd7ea015a5160d0afab55b837edf')
VERIFY = {'uniform-2': ('uniform', 2, 64), 'sgs-2': ('sgs', 2, 64),
          'manual-2': ('manual', 2, 64), 'sgs-20': ('sgs', 20, 4096)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def label_for(pool, sampler, seed):
    return f'p2b-formal-{pool}-R0-{sampler}-seed{seed}-u8000'


def verification_label(pool, case):
    return f'p2b-entry-{pool}-R0-{case}-seed43'


def config_for(pool, sampler, seed, machine, *, case=None):
    train = TRAINING[pool]
    return dict(SETTINGS, arm=sampler, sampler=sampler, student_seed=seed,
        environment_seed=seed, sampler_seed=seed, training_pool=train['path'],
        evaluation_pool=EVALUATION['path'], training_pool_content_sha256=train['content_sha256'],
        evaluation_pool_content_sha256=EVALUATION['content_sha256'],
        training_pool_file_sha256=train['file_sha256'], evaluation_pool_file_sha256=EVALUATION['file_sha256'],
        launch_status='formal_entry_validation' if case else 'formal_release_only',
        formal_machine=machine, main_pool=pool, verification_case=case,
        output_label=verification_label(pool, case) if case else label_for(pool, sampler, seed))


def schedule(pool):
    return [
        dict(machine=machine, seed=seed, sampler=arm, sequence=i+1, label=label_for(pool, arm, seed),
             config=f'configs/p2b_formal/{label_for(pool, arm, seed)}.json')
        for machine, seeds in SEEDS.items()
        for i, arm in enumerate(ARMS)
        for seed in seeds
    ]


def expected_execution(config, updates):
    settings = dict(reward='R0', initial_action_std=.3, gamma=.99, entropy_coef=.001,
        success_criterion='hold_only', student_seed=config['student_seed'],
        environment_seed=config['environment_seed'], sampler_seed=config['sampler_seed'],
        sampler=config['sampler'], rollout_steps=24, ema_decay=.99,
        success_reward=1000., physical_failure_reward=-100., timeout_penalty=-50.,
        success_radius_m=.05, hold_steps=25, timeout_steps=500, episode_seconds=10.,
        pools=dict(training=TRAINING[config['main_pool']], evaluation=EVALUATION),
        sgs=SGS if config['sampler'] == 'sgs' else None, evaluation_every_updates=100,
        raw_evaluation_every_updates=500, checkpoint_every_updates=500,
        configured_updates=8000, executed_updates=updates)
    return settings


def machine_gpu(machine):
    require(machine in GPU, 'Unknown formal machine')
    line = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader'],
        text=True, timeout=10).strip().splitlines()
    require(len(line) == 1, 'Formal admission expects the single assigned GPU')
    model, uuid, driver = (value.strip() for value in line[0].split(','))
    require(model == GPU[machine] and uuid.startswith('GPU-'), 'Wrong GPU for the assigned machine')
    return dict(model=model, uuid=uuid, driver=driver)


def load_release(root, expected_sha):
    require(isinstance(expected_sha, str) and re.fullmatch(r'[0-9a-f]{64}', expected_sha),
            'Pass the SHA256 of the Mac formal release manifest')
    path = root/RELEASE
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'Formal release manifest SHA256 differs')
    release = json.loads(raw)
    require(release.get('ready_for_formal_deployment') is True and release.get('base_commit') == BASE_COMMIT,
            'The formal package has not been released')
    require(re.fullmatch(r'[0-9a-f]{40}', release['git_commit']), 'Missing full release commit')
    decision_path = root/DECISION
    require(digest(decision_path) == release['decision_sha256'], 'Main-pool decision SHA256 differs')
    decision = read(decision_path)
    pool = release['main_pool']
    require(pool in TRAINING and decision['status'] == 'locked' and decision['main_pool'] == pool,
            'The main pool is not locked to this release')
    require(release['pools'] == dict(training=TRAINING[pool], evaluation=EVALUATION), 'Invalid pool bindings')
    for name, sha in release['files_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid release file path')
        require(digest(root/relative) == sha, 'Frozen source/config SHA256 differs: '+name)
    for item in release['pools'].values():
        require(digest(root/item['path']) == item['file_sha256'], 'Pool file SHA256 differs: '+item['path'])
    return release


def admit(root, config_path, config, verify, num_envs, out, device, machine, release_sha, acceptance_sha):
    release = load_release(root, release_sha)
    pool = release['main_pool']
    require(device == 'cuda:0' and machine in SEEDS, 'Invalid formal CUDA device or machine')
    require(not out.parent.is_symlink(), 'Formal output parent must be an independent directory')
    sampler, seed = config.get('sampler'), config.get('student_seed')
    case = config.get('verification_case')
    require(sampler in ARMS and type(seed) is int, 'Unknown formal arm or invalid seed')
    if verify is None:
        require(case is None and seed in SEEDS[machine] and num_envs == 4096, 'Wrong formal machine, seed or budget')
        expected = config_for(pool, sampler, seed, machine)
        entry = next(item for item in schedule(pool) if item['label'] == expected['output_label'])
        allowed = release['jobs']
        require(any(all(job.get(k) == v for k,v in entry.items()) for job in allowed), 'Job is outside the formal schedule')
        path = entry['config']
    else:
        require(case in VERIFY and VERIFY[case] == (sampler, verify, num_envs) and seed == 43,
                'Invalid verification arm, 2/20-update budget or environment count')
        expected = config_for(pool, sampler, seed, machine, case=case)
        path = release['verification'][machine][case]['config']
    require(config == expected, 'Formal config differs from frozen effective settings')
    require(config_path.resolve() == (root/path).resolve() and path in release['files_sha256'],
            'Formal config is outside the released path')
    output_root = (root if machine == 'yanglab' else root.parent)/'runs'
    wanted = output_root/expected['output_label']
    require(out.resolve() == wanted.resolve() and not out.exists() and not out.is_symlink(),
            'Output directory differs or the target already exists')
    gpu = machine_gpu(machine)
    if verify is None:
        accept = root/ACCEPTANCE
        require(isinstance(acceptance_sha, str) and digest(accept) == acceptance_sha,
                'Machine acceptance receipt SHA256 differs')
        receipt = read(accept)
        require(receipt['passed'] is True and receipt['release_sha256'] == release_sha
                and receipt['machine'] == machine and receipt['gpu'] == gpu
                and set(receipt['cases']) == set(VERIFY)
                and all(item['passed'] for item in receipt['cases'].values()),
                'This machine has not passed the released entry verification')
    else:
        require(acceptance_sha is None, 'The verification path cannot use a formal training receipt')
    return release, gpu
