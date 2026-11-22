"""Admit only the independent 32K Grid16 control authorized by sprint 19."""
import shlex
from pathlib import Path
import os

import numpy as np

from goalreach.p3s1_grid import GridSampler
from tools.p3b_protocol import checked
from tools.build_p3b_pool import require, sha256
from tools.p3s2_frozen_protocol import (HOSTS, LOCK, config_for_job, inspect_gpu, launcher_class, read)

REVISION = 'P3S2-middle-grid16-v1'
MIDDLE_PARENT = 'f25d1ff8ca9b2bbf74a263ee68b60e511f089c41c429af1a44d5ffb3e7c75a99'
YANG_CARD = dict(uuid=os.environ.get("NSGS_GPU_UUID_3", "unconfigured-GPU_UUID_3"), driver='535.230.02',
                 name='NVIDIA GeForce RTX 3060')
HOSTS = {'26957': HOSTS['26957'],
         'yanglab': dict(machine='yanglab', hostname='tongji-Precision-3680', cards={'0': YANG_CARD})}


def ordered_jobs(host):
    result = {}
    for index, seed in enumerate((46, 47) if host == '26957' else (43,)):
        alias = f'middle-grid16-{seed}'
        result[alias] = dict(alias=alias, kind='formal', scope='formal', seed=seed, gpu_index=index,
            gpu_uuid=HOSTS[host]['cards'][str(index)]['uuid'], arm='grid16_sharp', num_envs=4096,
            updates=8000, output=f'p3s2-{alias}-u8000', from_scratch=True, rollout_steps=24)
    return result


def grid_proof(pool_path):
    with np.load(pool_path, allow_pickle=False) as package:
        pool = {key: package[key] for key in ('region', 'horizontal_distance', 'azimuth', 'height')}
    sampler = GridSampler(pool, 0, 16)
    probabilities = sampler.probabilities()
    occupied = sampler.occupancy > 0
    cell_mass = np.bincount(sampler.ids, weights=probabilities, minlength=16**3)
    require(np.all(sampler.cell_rates == .5) and np.all(cell_mass[~occupied] == 0)
            and np.allclose(cell_mass, sampler.occupancy / sampler.n)
            and np.allclose(probabilities, 1 / sampler.n), 'Unexpected empty/unknown prior or pool masses')
    sampler.record([0], [1], [1])
    require(sampler.cell_rates[sampler.ids[0]] == 1 and np.all(sampler.cell_rates[~occupied] == .5),
            'Feedback leaked into empty cells')
    return dict(configurations=sampler.n, occupied_cells=int(occupied.sum()), empty_cells=int((~occupied).sum()),
                cell_ids_sha256=sampler.cell_ids_sha256, initial_prior=.5, initial_empty_mass=0.0,
                initial_unknown_occupied_mass=float(probabilities.sum()), sharp_target=.66)


def load(project, deployment_sha, *, verify_pools=True):
    deployment = checked(project / 'delivery/deployment.json', deployment_sha)
    host = deployment['host_key']
    parent = checked(project / 'delivery/p3s1_parent_deployment.json', MIDDLE_PARENT)
    require(deployment['revision'] == REVISION and deployment['method'] == 'middle'
            and deployment['host'] == HOSTS[host] and deployment['jobs'] == ordered_jobs(host)
            and deployment['pool'] == parent['pool'] and deployment['feature_binding'] == parent['feature_binding']
            and deployment['scorer_settings'] == parent['scorer_settings']
            and deployment['selection_sha256'] == LOCK
            and deployment['protocol_sha256'] == sha256(project / 'delivery/common_recipe.json')
            and deployment['main_pool'] == 'natural', 'Unexpected middle Grid16 deployment')
    for name, digest in deployment['files_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts
                and sha256(project / relative) == digest, 'Frozen file differs: ' + name)
    if verify_pools:
        for binding in deployment['pool'].values():
            require(sha256(project / binding['path']) == binding['file_sha256'], 'Pool bytes differ')
        require(grid_proof(project / deployment['pool']['training']['path']) == deployment['grid_proof'],
                'Middle pool grid mapping or prior differs')
    return deployment


def permit_for(project, deployment, deployment_sha, job, permit_sha):
    permit = checked(project / 'delivery' / f'permit_{job["alias"]}.json', permit_sha)
    require(permit['revision'] == REVISION and permit['deployment_sha256'] == deployment_sha
            and permit['allowed_job'] == job
            and permit['expected_initial_actor_sha256'] == deployment['initial_actors'][str(job['seed'])]
            and len(permit['expected_initial_actor_sha256']) == 64, 'Middle Grid16 permit differs')
    return permit


def admission(project, args):
    deployment = load(project, args.p3b_deployment_sha256)
    job = deployment['jobs'][args.p3s2_job]
    require(args.p3b_scope == job['scope'] and args.p3b_arm == job['arm']
            and args.p3b_machine == deployment['host']['machine'] and args.num_envs == job['num_envs']
            and args.headless and args.livestream == 0 and args.device == 'cuda:0' and args.throttle == 0
            and not any(getattr(args, key, False) for key in ('enable_cameras', 'xr', 'distributed')),
            'Middle Grid16 launch arguments differ')
    path = project / 'runs' / job['output']
    require(args.out.absolute() == args.out.resolve() == path and not path.exists() and not path.parent.is_symlink(),
            'Output has to be the fresh declared location')
    permit = permit_for(project, deployment, args.p3b_deployment_sha256, job, args.p3b_permit_sha256)
    card = inspect_gpu(deployment['host'], job)
    args.p3s2_renderer = int(os.environ['P3S2_KIT_GPU_INDEX'])
    require(args.p3s2_renderer in range(len(deployment['host']['cards'])), 'Unexpected renderer ordinal')
    denied = ('/renderer/activeGpu', '/physics/cudaDevice', '/renderer/multiGpu/')
    require(not any(any(key in part for key in denied) for part in shlex.split(args.kit_args)),
            'Runtime controls the rendering and physics bindings')
    runtime = {key: deployment[key] for key in ('pool', 'feature_binding', 'scorer_settings',
                                                'selection_sha256', 'protocol_sha256')}
    runtime.update(host=card, permit=permit, job=job, grid_bins=16)
    return runtime, config_for_job(deployment, job)


def verify_initial(deployment, config, digest, out):
    require(digest == deployment['permit']['expected_initial_actor_sha256'],
            'Middle Grid16 zero-start policy differs from its same-seed comparator')
