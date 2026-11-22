"""Frozen admission for the two sequential Yanglab neural-SGS simulator checks."""
import os
import ctypes
from pathlib import Path
import re
import socket
import subprocess
from uuid import UUID

from tools.p2b_K2_protocol import EVALUATION, TRAINING, config_for, digest, read, require

HOST = 'tongji-Precision-3680'
GPU = os.environ.get("NSGS_GPU_UUID_3", "unconfigured-GPU_UUID_3")
DRIVER = '535.230.02'
PACKAGE = 'delivery/p3_engineering_yanglab_package.json'
REMOTE = Path('/data/nsgs_goalreach_p3_engineering_20261115/simulator-r1')
KERNELS = ('default','sharp')
SEED = 43
UPDATES = 40
NUM_ENVS = 64
SETTINGS = dict(tau=100,alpha=None,features='raw',learning_rate=.0003)
SCORER = dict(members=5,hidden_layers=[256,256,256],steps_per_member_per_ppo=20,
              batch_size=4096,history_limit=1_000_000,cold_uniform_updates=20)


def label(kernel):
    return f'p3-yanglab-neural-natural-R0-{kernel}-seed43-u40'


def schedule():
    return [dict(kernel=kernel,run_id=label(kernel),seed=SEED,pool='natural',reward='R0',
                 environments=NUM_ENVS,updates=UPDATES,rollout_steps=24,from_scratch=True)
            for kernel in KERNELS]


def load(project, expected_sha):
    path=project/PACKAGE
    require(isinstance(expected_sha,str) and re.fullmatch('[0-9a-f]{64}',expected_sha)
            and digest(path)==expected_sha,'P3 Yanglab package SHA256 differs')
    manifest=read(path)
    require(manifest['scope']=='engineering_only' and manifest['ready_for_formal_training'] is False
            and manifest['simulator_host']==dict(hostname=HOST,gpu_uuid=GPU,driver=DRIVER)
            and manifest['remote_project']==str(REMOTE) and manifest['schedule']==schedule()
            and manifest['total_ppo_updates']==80 and manifest['sequential'] is True
            and manifest['staging_options_awaiting_2c']==SETTINGS and manifest['scorer']==SCORER,
            'Wrong P3 Yanglab host, engineering budget or scorer')
    for name,checksum in manifest['files_sha256'].items():
        member=Path(name)
        require(not member.is_absolute() and '..' not in member.parts
                and digest(project/member)==checksum,'P3 Yanglab frozen file changed: '+name)
    for binding in (TRAINING['natural'],EVALUATION):
        require(digest(project/binding['path'])==binding['file_sha256'],'P3 Yanglab pool binding differs')
    return manifest


def gpu_identity():
    cards=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],
                                 text=True,timeout=10).strip().splitlines()
    require(socket.gethostname()==HOST and len(cards)==1
            and [value.strip() for value in cards[0].split(',')]==[GPU,DRIVER],
            'P3 Yanglab host or NVIDIA GPU identity differs')


def visible_uuid():
    driver=ctypes.CDLL('libcuda.so.1')
    count=ctypes.c_int()
    device=ctypes.c_int()
    require(driver.cuInit(0)==0 and driver.cuDeviceGetCount(ctypes.byref(count))==0 and count.value==1
            and driver.cuDeviceGet(ctypes.byref(device),0)==0,
            'P3 Yanglab must expose exactly one physical CUDA GPU')
    return device_uuid(driver,device)


def active_uuid():
    driver=ctypes.CDLL('libcuda.so.1')
    device=ctypes.c_int()
    require(driver.cuCtxGetDevice(ctypes.byref(device))==0,'P3 CUDA context is inactive')
    return device_uuid(driver,device)


def device_uuid(driver,device):
    uuid=(ctypes.c_ubyte*16)()
    require(driver.cuDeviceGetUuid_v2(ctypes.byref(uuid),device)==0,'Cannot read P3 CUDA UUID')
    return 'GPU-'+str(UUID(bytes=bytes(uuid)))


def admission(project,args):
    manifest=load(project,args.p3_engineering_sha256)
    kernel=args.p3_yanglab_kernel
    require(kernel in KERNELS,'Only the fixed default and sharp engineering kernels are permitted')
    output=project/'runs'/label(kernel)
    require(args.headless and args.device=='cuda:0' and args.num_envs==NUM_ENVS and args.throttle==0
            and args.out.absolute()==output and args.out.resolve()==output
            and not output.exists() and not output.is_symlink() and not output.parent.is_symlink(),
            'P3 Yanglab requires the fixed 64-environment engineering output')
    gpu_identity()
    require(visible_uuid()==GPU,'P3 Yanglab CUDA UUID differs')
    config=config_for('natural','sgs_'+kernel,SEED,'p3-engineering-yanglab',scope='formal')
    config.update(updates=UPDATES,sampler='external',phase='3-engineering',num_envs=NUM_ENVS,
                  output_label=label(kernel),launch_status='engineering_only')
    return dict(manifest,staging_options_awaiting_2c=dict(SETTINGS,kernel=kernel)),config
