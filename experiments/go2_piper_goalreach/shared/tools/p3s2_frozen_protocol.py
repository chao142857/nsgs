"""Bounded P3S2 admission. Runtime loads only a host-specific released manifest."""
import csv
from pathlib import Path
import os
import shlex
import socket
import subprocess

from tools.build_p3b_pool import sha256, require
from tools.p2b_K2_protocol import config_for, read
from tools.p3b_protocol import checked
from tools.p3_yanglab_protocol import visible_uuid

DECISION_SHA='206f40e0f1faa3e9887af3d858db8968155ebbb7feb7bc69900f69f141a3acdd'
LOCK='bf28191e97855019ec5cdb20de1bfc4e6b08dd8ce70cd7104b4e5c5684a8fb67'
PARENTS={'grid16':'eb24bc6550f746ac77511905a774f40f3bd542b9f59f0bf713825dd50bf48aaa',
         'middle':'f25d1ff8ca9b2bbf74a263ee68b60e511f089c41c429af1a44d5ffb3e7c75a99'}
HOSTS={
    '4060ti':dict(machine='rental-4060ti',hostname=None,cards={
      '0':dict(uuid=os.environ.get("NSGS_GPU_UUID_4", "unconfigured-GPU_UUID_4"),
               name='NVIDIA GeForce RTX 4060 Ti',driver='550.127.08')}),
    '30817':dict(machine='rental-30817',hostname=os.environ.get("NSGS_HOSTNAME_1", "unconfigured-HOSTNAME_1"),cards={
      '0':dict(uuid=os.environ.get("NSGS_GPU_UUID_2", "unconfigured-GPU_UUID_2"),name='NVIDIA GeForce RTX 3060',driver='535.161.08')}),
    '26957':dict(machine='rental-dual3060-26957',hostname=os.environ.get("NSGS_HOSTNAME_5", "unconfigured-HOSTNAME_5"),cards={
      '0':dict(uuid=os.environ.get("NSGS_GPU_UUID_6", "unconfigured-GPU_UUID_6"),name='NVIDIA GeForce RTX 3060',driver='535.154.05'),
      '1':dict(uuid=os.environ.get("NSGS_GPU_UUID_7", "unconfigured-GPU_UUID_7"),name='NVIDIA GeForce RTX 3060',driver='535.154.05')})}
FAMILIES={'grid16':'grid16_sharp','middle':'neural_sharp'}
PROJECTIONS={'grid16':1048576,'middle':32768}


def jobs(host, method):
    arm=FAMILIES[method]
    data={}
    def put(alias, kind, seed, gpu, updates, selected_arm=arm, num_envs=4096):
        key='formal' if kind=='formal' else 'engineering'
        if kind.startswith('bridge_'):
            key='bridge'
        data[alias]=dict(alias=alias,kind=kind,scope=key,seed=seed,gpu_index=gpu,
            gpu_uuid=HOSTS[host]['cards'][str(gpu)]['uuid'],arm=selected_arm,updates=updates,num_envs=num_envs,
            output=f'p3s2-{alias}-u{updates}',from_scratch=True,rollout_steps=24,
            initialization_group=f'{host}/gpu{gpu}/seed{seed}/env{num_envs}')
    if method=='grid16':
        require(host in ('4060ti','30817'),'No registered grid host')
        seed=44 if host=='4060ti' else 48
        put(f'grid16-{host}-smoke','smoke',42,0,2,num_envs=64)
        put(f'grid16-{host}-accept','accept',42,0,200)
        put(f'million-grid16-{seed}','formal',seed,0,8000)
    else:
        require(host in ('4060ti','26957'),'No registered middle host')
        lanes=(0,) if host=='4060ti' else (0,1)
        for gpu in lanes:
            put(f'middle-{host}-g{gpu}-smoke','smoke',42,gpu,2,num_envs=64)
            put(f'middle-{host}-g{gpu}-accept','accept',42,gpu,200)
            if host=='26957':
                put(f'middle-26957-g{gpu}-parallel','parallel',42,gpu,100)
        for seed in ((44,45) if host=='4060ti' else (46,47)):
            gpu=0 if host=='4060ti' else seed-46
            for kind in ('bridge_old','bridge_new'):
                put(f'middle-{kind.replace("_","-")}-{seed}',kind,seed,gpu,20,'table_sharp')
            put(f'middle-neural-{seed}','formal',seed,gpu,8000)
    return data


def load(project, deployment_sha, *, verify_pools=True):
    project=project.resolve()
    deployment=checked(project/'delivery/deployment.json',deployment_sha)
    host,method=deployment['host_key'],deployment['method']
    require(deployment['revision']=='P3S2-v1' and deployment['decision_sha256']==DECISION_SHA
            and deployment['selection_sha256']==LOCK and deployment['parent_deployment_sha256']==PARENTS[method]
            and deployment['host']==HOSTS[host] and deployment['jobs']==jobs(host,method)
            and deployment['main_pool']=='natural', 'P3S2 deployment host, science or schedule differs')
    protocol=checked(project/'delivery/protocol.json',deployment['protocol_sha256'])
    require(protocol['decision_sha256']==DECISION_SHA and protocol['main_pool']=='natural'
            and protocol['engineering_max_ppo']==1370 and protocol['formal_max_ppo']==48000
            and protocol['schedule'][host][method]==deployment['jobs'],'P3S2 shared protocol differs')
    parent=checked(project/'delivery/p3s1_parent_deployment.json',PARENTS[method])
    require(sha256(project/'delivery/p3b_selection_v1.json')==LOCK and
            deployment['pool']==parent['pool'] and deployment['feature_binding']==parent['feature_binding']
            and deployment['scorer_settings']==parent['scorer_settings'], 'P3S2 parent pool, feature or scorer differs')
    require(deployment['pool']['training']['size']==PROJECTIONS[method]
            and deployment['pool']['evaluation']['size']==2016,'P3S2 pool sizes differ')
    for path,digest in deployment['files_sha256'].items():
        name=Path(path)
        require(not name.is_absolute() and '..' not in name.parts and sha256(project/name)==digest,
                'P3S2 frozen source differs: '+path)
    if verify_pools:
        for role,item in deployment['pool'].items():
            require(sha256(project/item['path'])==item['file_sha256'],'P3S2 wrong '+role+' pool bytes')
    return deployment


def inspect_gpu(host, job):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],
                                 text=True,timeout=12)
    cards={g.strip():dict(name=n.strip(),uuid=u.strip(),driver=v.strip())
           for g,n,u,v in csv.reader(rows.splitlines(),skipinitialspace=True)}
    require(cards==host['cards'] and (host['hostname'] is None or socket.gethostname()==host['hostname'])
            and os.environ.get('CUDA_VISIBLE_DEVICES')==job['gpu_uuid'] and visible_uuid()==job['gpu_uuid'],
            'P3S2 physical inventory or CUDA lane differs')
    return cards[str(job['gpu_index'])]


def permit_for(project, deployment, deploy_sha, job, permit_sha):
    require(job['kind']=='formal' and permit_sha is not None,'Only one signed formal alias may start')
    path=project/'delivery'/f'permit_{job["alias"]}.json'
    permit=checked(path,permit_sha)
    acceptance=checked(project/'delivery'/'acceptance.json',permit['acceptance_sha256'])
    require(permit['revision']=='P3S2-permit-v1' and permit['deployment_sha256']==deploy_sha
            and permit['protocol_sha256']==deployment['protocol_sha256'] and permit['decision_sha256']==DECISION_SHA
            and permit['method']==deployment['method'] and permit['host_key']==deployment['host_key']
            and permit['allowed_job']==job and permit['gpu_uuid']==job['gpu_uuid']
            and len(permit['expected_initial_actor_sha256'])==64
            and acceptance['deployment_sha256']==deploy_sha and acceptance['passed'] is True,
            'P3S2 formal permit or machine acceptance differs')
    if deployment['method']=='middle':
        bridge=checked(project/'delivery'/'baseline_reuse.json',permit['baseline_reuse_sha256'])
        entry=bridge['seeds'][str(job['seed'])]
        require(entry['reuse'] is True and entry['actor_sha256']==permit['expected_initial_actor_sha256']
                and entry['gpu_uuid']==job['gpu_uuid'] and entry['comparison']['bitwise_policy_optimizer_ema_rng'],
                'P3S2 bridge lacks a verified same-GPU old baseline')
    return permit


def config_for_job(deployment, job):
    method=deployment['method']
    config=config_for('natural','sgs_sharp',job['seed'],deployment['host']['machine'])
    config.update(phase='P3S2',protocol_revision='P3S2-v1',arm_id=job['arm'],launch_status=job['scope'],
        updates=job['updates'],num_envs=job['num_envs'],sampler='external' if job['arm']=='neural_sharp' else 'sgs',
        output_label=job['output'],raw_evaluation_every_updates=500 if job['scope']=='formal' else 200,
        checkpoint_every_updates=500 if job['scope']=='formal' else 200)
    for role,item in deployment['pool'].items():
        config.update({role+'_pool':item['path'],role+'_pool_size':item['size'],
                       role+'_pool_content_sha256':item['content_sha256'],role+'_pool_file_sha256':item['file_sha256']})
    require(config['main_pool']=='natural' and config['training_pool_size']==PROJECTIONS[method],
            'P3S2 main_pool mode or size differs')
    return config


def admission(project,args):
    sha=args.p3b_deployment_sha256
    deployment=load(project,sha)
    job=deployment['jobs'][args.p3s2_job]
    require(args.p3b_scope==job['scope'] and args.p3b_arm==job['arm']
            and args.p3b_machine==deployment['host']['machine'] and args.num_envs==job['num_envs']
            and args.headless and args.livestream==0 and args.device=='cuda:0' and args.throttle==0
            and not any(getattr(args,k,False) for k in ('enable_cameras','xr','distributed')),
            'P3S2 launch arguments differ from job')
    require(bool(args.p3b_permit_sha256)==(job['kind']=='formal'),'P3S2 requires its own formal permit')
    path=project/'runs'/job['output']
    require(args.out.absolute()==args.out.resolve()==path and not path.exists() and not path.parent.is_symlink(),
            'P3S2 output must be a fresh declared directory')
    permit=permit_for(project,deployment,sha,job,args.p3b_permit_sha256) if job['kind']=='formal' else None
    host=inspect_gpu(deployment['host'],job)
    args.p3s2_renderer=int(os.environ.get('P3S2_KIT_GPU_INDEX',str(job['gpu_index'])))
    require(args.p3s2_renderer in range(len(deployment['host']['cards'])), 'Unexpected Kit ordinal')
    denied=('/renderer/activeGpu','/physics/cudaDevice','/renderer/multiGpu/')
    require(not any(any(k in part for k in denied) for part in shlex.split(args.kit_args)),
            'P3S2 launch sets render and physics bindings itself')
    runtime=dict(pool=deployment['pool'],feature_binding=deployment['feature_binding'],
                 scorer_settings=deployment['scorer_settings'],selection_sha256=deployment['selection_sha256'],
                 protocol_sha256=deployment['protocol_sha256'],host=host,permit=permit,job=job,
                 grid_bins=16 if deployment['method']=='grid16' else None)
    return runtime,config_for_job(deployment,job)


def verify_initial(deployment,config,digest,out):
    permit=deployment['permit']
    if permit:
        require(digest==permit['expected_initial_actor_sha256'],'P3S2 from-zero policy differs from same-seed comparator')


def launcher_class(base, renderer):
    class BoundLauncher(base):
        def _resolve_device_settings(self, config):
            super()._resolve_device_settings(config)
            config['active_gpu']=renderer
            config['physics_gpu']=0
            config['multi_gpu']=False
    return BoundLauncher
