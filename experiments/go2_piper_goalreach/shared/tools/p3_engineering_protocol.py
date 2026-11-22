"""Admit only the explicitly packaged, bounded neural integration check."""
import os
from pathlib import Path
import re
import socket
import subprocess

from tools.p2b_K2_protocol import EVALUATION, TRAINING, config_for, digest, read, require

HOST = os.environ.get("NSGS_HOSTNAME_1", "unconfigured-HOSTNAME_1")
GPU = os.environ.get("NSGS_GPU_UUID_2", "unconfigured-GPU_UUID_2")
DRIVER = '535.161.08'
SEED = 43
LABEL = 'p3-engineering-natural-R0-default-seed43-u40'
MAX_UPDATES = 40
NUM_ENVS = 64
SETTINGS = dict(kernel='default',tau=100,alpha=None,features='raw',learning_rate=.0003)
PACKAGE = 'delivery/p3_engineering_package.json'


def admission(root, args):
    require(isinstance(args.p3_engineering_sha256,str) and re.fullmatch('[a-f0-9]{64}',args.p3_engineering_sha256)
            and digest(root/PACKAGE)==args.p3_engineering_sha256,'P3 engineering package SHA256 differs')
    receipt=read(root/PACKAGE)
    require(receipt['scope']=='engineering_only' and receipt['ready_for_formal_training'] is False
            and receipt['simulator_probe']==dict(max_updates=MAX_UPDATES,num_envs=NUM_ENVS,seed=SEED,gpu_uuid=GPU)
            and receipt['staging_options_awaiting_2c']==SETTINGS,'Wrong P3 engineering scope')
    for name, checksum in receipt['files_sha256'].items():
        path=Path(name)
        require(not path.is_absolute() and '..' not in path.parts and digest(root/path)==checksum,
                'P3 engineering package file changed')
    require(args.headless and args.device=='cuda:0' and args.num_envs==NUM_ENVS and args.throttle==0
            and args.out.absolute().resolve()==(root.parent/'runs'/LABEL).resolve()
            and not args.out.exists() and not args.out.is_symlink()
            and not args.out.parent.is_symlink(),'P3 engineering GPU or output settings differ')
    card=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,driver_version','--format=csv,noheader'],
                                 text=True,timeout=10).strip()
    require(socket.gethostname()==HOST and [x.strip() for x in card.split(',')]==[GPU,DRIVER],
            'P3 simulator check requires the assigned single GPU')
    for binding in (TRAINING['natural'],EVALUATION):
        require(digest(root/binding['path'])==binding['file_sha256'],'P3 pool binding differs')
    config=config_for('natural','sgs_default',SEED,'p3-engineering',scope='formal')
    config.update(updates=MAX_UPDATES, sampler='external', phase='3-engineering',
                  num_envs=NUM_ENVS,output_label=LABEL, launch_status='engineering_only')
    return receipt,config
