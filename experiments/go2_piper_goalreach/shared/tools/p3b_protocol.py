"""P3B1 deployment, scope, hardware identity, and from-zero entry checks."""
import os
import hashlib
from pathlib import Path
import re
import socket
import subprocess

from goalreach.sgs_kernels import KERNELS
from tools.build_p3b_pool import COUNTS, EVAL_SIZE, NEAR, SEED, SIZE, require, sha256
from tools.p2b_K2_protocol import config_for, read
from tools.p3_yanglab_protocol import visible_uuid

DEPLOYMENT='delivery/p3b_P3B1_deployment.json'
SELECTION='delivery/p3b_selection_v1.json'
POOL='delivery/p3b_P3B1_pool_manifest.json'
ARMS=('table_sharp','neural_sharp')
HOSTS={
    'yanglab':dict(model='NVIDIA GeForce RTX 3060',uuid=os.environ.get("NSGS_GPU_UUID_3", "unconfigured-GPU_UUID_3"),
                   driver='535.230.02',hostname='tongji-Precision-3680',formal_seed=43),
    'rental-4060ti':dict(model='NVIDIA GeForce RTX 4060 Ti',uuid=os.environ.get("NSGS_GPU_UUID_4", "unconfigured-GPU_UUID_4"),
                        driver='550.127.08',hostname=None,formal_seed=44),
}
ACTORS='initial_training_state.json'
FEATURE_CONTRACT='p2c_G5'


def slug(scope,machine,arm):
    seed=42 if scope=='engineering' else HOSTS[machine]['formal_seed']
    updates=200 if scope=='engineering' else 8000
    return f'p3b-P3B1-{scope}-{machine}-{arm}-seed{seed}-u{updates}'


def schedule():
    return {machine:{scope:[dict(arm=arm,seed=42 if scope=='engineering' else host['formal_seed'],
                updates=200 if scope=='engineering' else 8000,output=slug(scope,machine,arm),
                num_envs=4096,rollout_steps=24,from_scratch=True) for arm in ARMS]
            for scope in ('engineering','formal')} for machine,host in HOSTS.items()}


def settings(selection):
    online=selection['online']
    neural=online['neural_arm']
    require(online['kernel_id']=='sharp' and online['kernel']==dict(KERNELS['sharp'])
        and online['additional_uniform_mixing']==0 and online['table_arm']['window_natural_outcomes_per_task']==100
        and online['table_arm']['estimator']=='H100' and online['table_arm']['unseen_prior']==.5,'K2 sharp choice differs')
    require((neural['estimator'],neural['alpha'],neural['members'],neural['hidden_layers'],
             neural['steps_per_member_per_ppo'],neural['batch_size'],neural['history_limit'],neural['cold_uniform_updates'],
             neural['activation'],neural['cold_start_prior'],neural['learning_rate'])==
             ('pure E4',None,5,[256,256,256],20,4096,1000000,20,'ReLU',.5,.0003)
        and neural['optimizer']==dict(name='Adam',learning_rate=.0003,betas=[.9,.999],epsilon=1e-8,
                                     weight_decay=0.,amsgrad=False),'Online optimizer/budget differs')
    recipe=neural['feature_recipe']
    require(recipe['name']==neural['features'] and recipe['base_columns']==['target_x','target_y','target_z',
        'horizontal_distance','sin_azimuth','cos_azimuth','nominal_distance','episode_seconds']
        and recipe['standardization']['arithmetic']=='float64 after assembling raw columns; population mean and SD (ddof=0)'
        and recipe['standardization']['fit_on']=='entire frozen 1048576-task P3B1 training pool, once in canonical row order'
        and recipe['standardization']['evaluation_labels_or_rows_used'] is False, 'P3B1 feature geometry differs')
    if neural['features']=='fourier':
        require(recipe['fourier']['frequency_multipliers']==[1,2]
                and recipe['fourier']['append_order']==['sin(pi*z)','cos(pi*z)','sin(2*pi*z)','cos(2*pi*z)'],
                'Fourier channels differ')
    require(online['training_pool']==dict(kind='natural',size=SIZE,generator_seed=SEED,counts=list(COUNTS),
        near_distance_band_counts=list(NEAR)) and online['evaluation_pool']['size']==EVAL_SIZE
        and online['evaluation_pool']['cells']==42, 'Fixed pool settings differ')
    return dict(kernel='sharp',tau=neural['tau'],alpha=None,features=neural['features'],
                feature_contract=FEATURE_CONTRACT,learning_rate=neural['learning_rate'],adam_epsilon=neural['optimizer']['epsilon'])


def checked(path, expected):
    require(isinstance(expected,str) and re.fullmatch('[0-9a-f]{64}',expected) and sha256(path)==expected,
            f'Frozen SHA256 differs: {path}')
    return read(path)


def load(project, expected):
    deploy=checked(project/DEPLOYMENT,expected)
    require(deploy['revision']=='P3B1-v1' and deploy['scope']=='P3B1-deployment'
        and deploy['hosts']==HOSTS and deploy['schedule']==schedule(), 'Wrong deployment scope/hosts/budget')
    for name,digest in deploy['files_sha256'].items():
        path=Path(name)
        require(not path.is_absolute() and '..' not in path.parts and sha256(project/path)==digest,
                f'Frozen deployment file differs: {name}')
    lock=checked(project/SELECTION,deploy['selection_sha256'])
    pool=checked(project/POOL,deploy['pool_manifest_sha256'])
    require(deploy['scorer_settings']==settings(lock) and pool['seed']==SEED and
            pool['training']['size']==SIZE and pool['evaluation']['size']==EVAL_SIZE, 'Selection/pool binding differs')
    for key in ('file_sha256','content_sha256'):
        require(lock['online']['evaluation_pool'][key]==pool['evaluation'][key],'Fixed evaluation binding differs')
    for role in ('training','evaluation'):
        item=pool[role]
        require(sha256(project/item['path'])==item['file_sha256'],f'P3B1 {role} pool file differs')
    return dict(deploy,pool=pool)


def actual_gpu(machine):
    host=HOSTS[machine]
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version',
                                  '--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    allowed=[[row.strip() for row in line.split(',')] for line in rows]
    require([host[k] for k in ('model','uuid','driver')] in allowed
        and (not host['hostname'] or socket.gethostname()==host['hostname']), 'Physical host/GPU differs')
    require(visible_uuid()==host['uuid'], 'Assigned CUDA-visible UUID differs')
    return host


def verify_permit(project,args,deploy):
    machine=args.p3b_machine
    permit=checked(project/'delivery'/f'p3b_P3B1_permit_{machine}.json',args.p3b_permit_sha256)
    acceptance=checked(project/'delivery'/f'p3b_P3B1_acceptance_{machine}.json',permit['acceptance_sha256'])
    require(permit['scope']=='formal' and permit['machine']==machine and permit['seed']==HOSTS[machine]['formal_seed']
        and permit['deployment_sha256']==args.p3b_deployment_sha256 and permit['schedule']==schedule()[machine]['formal']
        and acceptance['scope']=='engineering' and acceptance['machine']==machine and acceptance['passed'] is True
        and acceptance['gpu_uuid']==HOSTS[machine]['uuid'] and acceptance['deployment_sha256']==args.p3b_deployment_sha256,
        'Formal permit or machine-local acceptance differs')
    return permit


def admission(project,args):
    machine, scope, arm=args.p3b_machine,args.p3b_scope,args.p3b_arm
    require(scope in ('engineering','formal') and machine in HOSTS and arm in ARMS
        and args.headless and args.device=='cuda:0' and args.num_envs==4096 and args.throttle==0,
        'P3B1 requires one scoped 4096-environment CUDA entry')
    require(bool(args.p3b_permit_sha256)==(scope=='formal'),'Only formal runs require a signed machine permit')
    output=project/'runs'/slug(scope,machine,arm)
    require(args.out.absolute()==output and args.out.resolve()==output and not output.exists()
            and not output.is_symlink() and not output.parent.is_symlink(),'Invalid P3B1 output or existing run')
    deploy=load(project,args.p3b_deployment_sha256)
    host=actual_gpu(machine)
    permit=verify_permit(project,args,deploy) if scope=='formal' else None
    entry=next(row for row in deploy['schedule'][machine][scope] if row['arm']==arm)
    config=config_for('natural','sgs_sharp',entry['seed'],machine)
    config.update(phase='P3B1',protocol_revision='P3B1-v1',arm_id=arm,launch_status=scope,
        updates=entry['updates'],sampler='sgs' if arm=='table_sharp' else 'external',output_label=entry['output'],
        raw_evaluation_every_updates=200 if scope=='engineering' else 500,
        checkpoint_every_updates=200 if scope=='engineering' else 500,
        training_pool_size=SIZE,evaluation_pool_size=EVAL_SIZE)
    for role in ('training','evaluation'):
        item=deploy['pool'][role]
        config.update({role+'_pool':item['path'],role+'_pool_content_sha256':item['content_sha256'],
                       role+'_pool_file_sha256':item['file_sha256']})
    return dict(deploy,host=host,permit=permit,machine=machine),config


def verify_initial(deploy,config,digest,out):
    seed=config['student_seed']
    machine=deploy['machine']
    scope=config['launch_status']
    if deploy['permit'] and deploy['permit'].get('expected_initial_actor_sha256'):
        require(digest==deploy['permit']['expected_initial_actor_sha256'], 'Formal initial actor differs from its seed binding')
    if config['arm_id']=='neural_sharp':
        peer=out.parent/slug(scope,machine,'table_sharp')
        initial=read(peer/ACTORS)
        result=read(peer/'result.json')
        require(initial['actor_content_sha256']==digest and initial['student_seed']==seed
                and result['completed_updates']==config['updates'] and result['passed'] is True,
                'Peer arm did not complete with the same from-zero actor')


def training_state_digest(runtime):
    # Raw ndarray bytes avoid JSON object overhead for million-entry state.
    checksum=hashlib.sha256()
    state=runtime.sampler.state_dict()
    for key,value in state.items():
        checksum.update(key.encode())
        checksum.update(value.tobytes() if hasattr(value,'tobytes') else repr(value).encode())
    for key,value in runtime.forecast.state_dict().items():
        checksum.update(key.encode())
        checksum.update(value.tobytes() if hasattr(value,'tobytes') else repr(value).encode())
    if runtime.neural:
        checksum.update(runtime.neural.digest().encode())
    return checksum.hexdigest()
