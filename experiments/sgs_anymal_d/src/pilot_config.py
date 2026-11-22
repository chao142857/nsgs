"""Run identity and crash-safe PPO budget for the bounded single-GPU pilot."""
from contextlib import contextmanager
from hashlib import sha256
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess

CAP = 500
CLASS_NAMES = ('birth_near', 'same_platform', 'obstacle')
SOURCE_FILES = ('train.py', 'pilot_config.py', 'isaac_entry.py', 'native.py', 'scene.py',
                'diagnostics.py', 'course.py', 'base_course.py', 'terrain.py', 'terrain_compat.py',
                'navigation.py', 'sampler.py', 'table.py', 'ppo_smoke.py', 'catalog.py')


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)


def file_sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def git_commit(root):
    supplied = os.environ.get('SGS_GIT_COMMIT')
    try:
        actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', '--verify', 'HEAD'],
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        actual = supplied
    if not actual or not re.fullmatch(r'[0-9a-f]{40}', actual):
        raise ValueError('Use a git checkout or set SGS_GIT_COMMIT to the exported full commit SHA')
    if supplied and actual != supplied:
        raise ValueError('SGS_GIT_COMMIT differs from checkout')
    return actual


def identity(root, args):
    import numpy as np
    root = Path(root)
    protocol = read(root/'protocol.json')
    if protocol['pilot_update_cap'] != CAP or protocol['num_envs'] != 4096:
        raise ValueError('Protocol does not describe the bounded pilot')
    if not 0 < args.updates <= CAP or not 0 < args.num_envs <= protocol['num_envs']:
        raise ValueError('Expected at most 500 updates and 4,096 environments')
    if args.eval_at != sorted(set(args.eval_at)) or any(u < 1 or u > args.updates for u in args.eval_at):
        raise ValueError('Evaluation nodes must be increasing and within this run')
    if args.updates not in args.eval_at:
        raise ValueError('Include the last update as an evaluation node')
    meta = read(root/'catalog.json')
    geom = []
    for name in ('geometry.npy', 'holdout.npy'):
        a = np.load(root/name, allow_pickle=False)
        if a.shape != (104000,7) or a.dtype != np.float32 or not np.isfinite(a).all():
            raise ValueError(f'Invalid {name}')
        geom.append(sha256(a.tobytes(order='C')).hexdigest())
    if geom != meta['geometry_sha256']:
        raise ValueError('Coordinates differ from the generated catalog')
    file_hash = {}
    for split in ('train','holdout'):
        yaw = np.load(root/f'{split}_yaw.npy', allow_pickle=False)
        classes = np.load(root/f'{split}_classes.npy', allow_pickle=False)
        if yaw.shape != (104000,) or not np.isfinite(yaw).all() or not np.all(yaw.reshape(-1,20)==yaw.reshape(-1,20)[:,:1]):
            raise ValueError(f'Invalid or goal-dependent {split} birth yaw')
        if classes.shape != yaw.shape or classes.dtype != np.uint8 or np.any(classes>2):
            raise ValueError(f'Invalid {split} route classes')
        routes = np.load(root/f'{split}_route_types.npy', allow_pickle=False)
        if routes.shape != yaw.shape or routes.dtype != np.uint8 or np.any(routes>=len(meta['route_types'])):
            raise ValueError(f'Invalid {split} fine route labels')
        for suffix in ('yaw','classes','route_types'):
            name=f'{split}_{suffix}.npy'
            file_hash[name]=file_sha(root/name)
    item = dict(schema=2, git_commit=git_commit(root), geometry_hashes=geom, route_classes_sha256=file_hash,
                config_sha256=file_sha(root/'protocol.json'), catalog_sha256=file_sha(root/'catalog.json'),
                fixed_routes_sha256=file_sha(root/'fixed_routes.json'),
                source_sha256={n:file_sha(root/'src'/n) for n in SOURCE_FILES},
                num_envs=args.num_envs, updates=args.updates, eval_at=args.eval_at, rollout_controls=24, seed=args.seed)
    item['manifest']=sha256(json.dumps(item,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return item


def check_physics(path, run):
    receipt=read(path)
    if not (receipt.get('passed') is True and receipt.get('zero_ppo') is True
            and receipt.get('geometry_hashes')==run['geometry_hashes']
            and receipt.get('git_commit')==run['git_commit']):
        raise ValueError('Physical acceptance must pass on this commit and pair of banks')
    return file_sha(path)


@contextmanager
def locked_budget(path, run, out, resume):
    """A failed or interrupted optimizer call also consumes a reserved PPO update."""
    path=Path(path).resolve()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_name(path.name+'.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
        if path.exists():
            budget=read(path)
            if budget['manifest']!=run['manifest'] or budget['out']!=str(out.resolve()):
                raise RuntimeError('Budget already belongs to another run')
            if budget['reserved'] and not resume:
                raise RuntimeError('Use --resume-at with the latest checkpoint')
        else:
            if resume:
                raise RuntimeError('Cannot recover without the original budget ledger')
            budget={'cap':CAP,'manifest':run['manifest'],'out':str(out.resolve()),'reserved':0,'completed':0}
            atomic_json(path,budget)
        yield path,budget


def reserve(path, budget):
    if budget['reserved']>=budget['cap']:
        return False
    budget['reserved']+=1
    atomic_json(path,budget)
    return True
