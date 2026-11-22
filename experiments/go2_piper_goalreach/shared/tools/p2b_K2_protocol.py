"""K2 verification, provisional and formal admission before Isaac initialization."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math

from goalreach.sgs_kernels import KERNELS, EMPTY_HISTORY_PRIOR
from tools.p2b_formal_protocol import (TRAINING, EVALUATION, SEEDS, config_for as prior_config,
    digest, expected_execution as prior_execution, machine_gpu, read, require)
from tools.p2b_pool_decision import EFFECTIVE

REVISION = 'K2'
RELEASE_ID = 'p2b-K2-20261114-r2'
ARMS = ('uniform','manual','sgs_default','sgs_sharp')
BATCHES = ((('uniform',44),('uniform',45)), (('sgs_default',44),('sgs_sharp',45)),
           (('sgs_sharp',44),('sgs_default',45)), (('manual',44),('manual',45)))
DEPLOY = 'delivery/p2b_K2_deployment.json'
PROVISIONAL = 'delivery/p2b_K2_provisional_release.json'
FORMAL = 'delivery/p2b_K2_formal_release.json'
ADOPTION = 'delivery/p2b_K2_adoption.json'
STATUS = 'delivery/p2b_K2_handoff_status.json'
DECISION = 'delivery/p2b_main_pool_decision.json'
ACCEPTANCE = 'runtime/p2b_K2_entry_acceptance.json'
SCOPES = {'gpu_acceptance':DEPLOY,'provisional_natural_uniform':PROVISIONAL,'formal':FORMAL}
PILOT_END = '/data/nsgs_goalreach_p2b_20261114/runtime/p2b_pilot_status.json'
SHORT = {**{f'{arm}-2':(arm,2,64) for arm in ARMS},
         'sgs_sharp-20':('sgs_sharp',20,4096),'sgs_sharp-200':('sgs_sharp',200,4096),
         'sgs_default-20':('sgs_default',20,4096),
         'sgs_default-100':('sgs_default',100,4096),'sgs_sharp-100':('sgs_sharp',100,4096)}
REPORTING = dict(training='Noisy natural terminal results with counts, censoring, action std and physical failure',
    evaluation='Deterministic EMA every 100; raw every 500; complete common task set',
    tail_updates=list(range(7000,8001,100)), slope_per_updates=1000,
    endpoints=[7800,7900,8000], head_of_tail=[7000,7100,7200], auc_budget=8000,
    paired_contrasts=['sgs_default-uniform','sgs_sharp-uniform','sgs_sharp-sgs_default'])


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def config_for(pool, arm, seed, machine, *, scope='formal', case=None):
    kernel = arm[4:] if arm.startswith('sgs_') else None
    data = prior_config(pool, arm, seed, machine)
    prefix = {'formal':'p2b-formal','provisional_natural_uniform':'p2b-provisional',
              'gpu_acceptance':'p2b-K2-verify'}[scope]
    label = f'{prefix}-{pool}-R0-{case or arm}-seed{seed}-u{SHORT[case][1] if case else 8000}'
    return dict(data, protocol_revision=REVISION, arm_id=arm, kernel_id=kernel,
                sampler='sgs' if kernel else arm, sgs=dict(KERNELS[kernel]) if kernel else None,
                verification_case=case, output_label=label, launch_status=scope)


def execution(config, updates):
    return dict(prior_execution(config,updates), protocol_revision=REVISION, arm_id=config['arm_id'],
                kernel_id=config['kernel_id'], sgs=config['sgs'],
                empty_history_prior=EMPTY_HISTORY_PRIOR if config['kernel_id'] else None)


def jobs_for(pool, scope):
    if scope == 'formal':
        jobs = [('yanglab',43,arm,i) for i,arm in enumerate(ARMS,1)]
        jobs += [('rental-4060ti',seed,arm,i) for i,batch in enumerate(BATCHES,1) for arm,seed in batch]
    elif scope == 'provisional_natural_uniform':
        require(pool=='natural','Provisional requires natural pool')
        jobs = [('rental-4060ti',seed,'uniform',1) for seed in (44,45)]
    else:
        raise ValueError('Unknown long-training scope')
    return [entry(config_for(pool,arm,seed,machine,scope=scope),batch=batch)
            for machine,seed,arm,batch in jobs]


def entry(config, *, batch=None):
    pool, machine, arm, label = (config[k] for k in ('main_pool','formal_machine','arm_id','output_label'))
    case = config['verification_case']
    updates, envs = SHORT[case][1:] if case else (8000,4096)
    path = f'configs/p2b_K2/{config["launch_status"]}/{machine}/{label}.json'
    out = f'runs/{label}' if machine == 'yanglab' else f'../runs/{label}'
    args = ['tools/run.py','--mode','p2b','--p2b-config',path,'--device','cuda:0','--throttle','0',
            '--num-envs',str(envs),'--out',out,'--p2b-formal-machine',machine,
            '--p2b-K2-scope',config['launch_status'],'--p2b-formal-release-sha256','<release_sha256>']
    args += ['--p2b-verify-updates',str(updates)] if case else [
             '--p2b-formal-acceptance-sha256','<machine_acceptance_sha256>']
    return dict(machine=machine, pool=pool, arm_id=arm, sampler=config['sampler'], seed=config['student_seed'],
                sequence=batch,label=label,config=path,case=case,updates=updates,num_envs=envs,
                execution_settings=execution(config,updates),runner_args=args)


def verification():
    return [entry(config_for(pool,arm,43,machine,scope='gpu_acceptance',case=case))
            for pool in TRAINING for machine in SEEDS for case,(arm,_,_) in SHORT.items()
            if machine!='yanglab' or not case.endswith('-100')]


def checked_reference(root, item):
    path = root/item['path']
    require(digest(path)==item['sha256'],'Referenced SHA256 differs: '+str(path))
    return read(path)


def validate_provisional(a4, a3):
    require(a4.get('protocol_revision')==a3.get('protocol_revision')==REVISION,
            'Provisional requires K2 producer receipts')
    require(a4.get('status')=='complete' and a4.get('valid') is True and a4.get('arm')=='A4'
            and a4.get('host')=='ubuntu-robot' and a4.get('completed_updates')==a4.get('budget')==8000
            and a4.get('effective_configuration')==EFFECTIVE,'Expected a verified natural A4')
    require(a4.get('pool')==a3.get('pool')=='natural', 'Wrong provisional pool')
    for role,binding in (('train',TRAINING['natural']),('evaluation',EVALUATION)):
        item=a4['pools'][role]
        require(Path(item['path']).as_posix().endswith('/'+binding['path']) and
                all(item[k]==binding[k] for k in ('content_sha256','file_sha256')), 'Wrong A4 pool binding')
    rows=a4['last_three_ema']
    require([r['update'] for r in rows]==[7800,7900,8000]
        and all(r['tasks']==1008 and r['task_count']==2016 and r['cell_count']==42 and
                type(r['successes']) is int and 0<=r['successes']<=1008 for r in rows)
        and sum(r['successes'] for r in rows)>=1512
        and math.isclose(a4['mean'],sum(r['successes'] for r in rows)/3024,abs_tol=1e-12),
        'A4 walk gate is below 50% or incomplete')
    require(a3.get('status')=='deferred' and a3.get('host')=='ubuntu-robot' and a3.get('arm')=='A3'
            and a3.get('triggered') is True and a3.get('budget')==8000,
            'A3 has no explicit original-window deferral')
    budget=float(a3['unadjusted_estimate_s'])
    padded=float(a3['padded_estimate_s'])
    estimated=datetime.fromisoformat(a3['estimated_finish'])
    checked=datetime.fromisoformat(a3['window_checked_at'])
    hour,minute=map(int,a3['latest_allowed_finish_local'].split(':'))
    cutoff=checked.replace(hour=hour,minute=minute,second=0,microsecond=0)
    if cutoff < checked:
        cutoff+=timedelta(days=1)
    earliest=datetime.fromisoformat(a3['earliest_allowed_start'])
    require(budget>0 and math.isclose(padded,1.15*budget,abs_tol=1.) and
            math.isclose((estimated-checked).total_seconds(),padded,abs_tol=2.) and
            estimated>cutoff and earliest>cutoff,'A3 scheduling evidence does not establish deferral')


def load_release(root, expected_sha, scope):
    require(scope in SCOPES,'K2 scope required; the previous schedule is retired')
    path=root/SCOPES[scope]
    require(expected_sha and digest(path)==expected_sha,'K2 release SHA256 differs')
    release=read(path)
    require(release.get('protocol_revision')==REVISION and release.get('scope')==scope,'Wrong protocol revision or scope')
    deploy=release if scope=='gpu_acceptance' else checked_reference(root,release['deployment'])
    require(deploy['protocol_revision']==REVISION and deploy['scope']=='gpu_acceptance'
            and deploy['ready_for_gpu_acceptance'] is True
            and deploy['git_commit']==release['git_commit']
            and deploy['calculation_sha256']==release['calculation_sha256'],'Wrong K2 deployment binding')
    for relative, checksum in {**deploy['files_sha256'],**release['files_sha256']}.items():
        name=Path(relative)
        require(not name.is_absolute() and '..' not in name.parts and digest(root/name)==checksum,
                'Frozen K2 source/config SHA256 differs: '+relative)
    pool=release.get('main_pool')
    if scope=='formal':
        decision=checked_reference(root,release['decision'])
        require(release.get('ready_for_formal_deployment') is True and decision['status']=='locked'
                and decision['main_pool']==pool and release['jobs']==jobs_for(pool,scope),'Formal schedule or main pool not locked')
    elif scope=='provisional_natural_uniform':
        require(release.get('ready_for_provisional_training') is True and pool=='natural'
            and release.get('ready_for_formal_deployment') is False
            and release['jobs']==jobs_for('natural',scope),'Wrong provisional schedule')
        validate_provisional(*(checked_reference(root,release['evidence'][k]) for k in ('a4','a3')))
    return release,deploy


def required_cases(machine, scope):
    if scope=='provisional_natural_uniform':
        return {'uniform-2'}
    cases={f'{arm}-2' for arm in ARMS}|{'sgs_sharp-20','sgs_sharp-200'}
    return cases|({'dual-default-sharp-100','sharp-first20-equivalence'} if machine=='rental-4060ti' else set())


def validate_acceptance(root, receipt_sha, machine, gpu, pool, scope, deploy, deploy_sha):
    path=root/ACCEPTANCE
    require(receipt_sha and digest(path)==receipt_sha,'K2 machine receipt SHA256 differs')
    receipt=read(path)
    require(receipt.get('protocol_revision')==REVISION and receipt.get('passed') is True
        and receipt.get('machine')==machine and receipt.get('gpu')==gpu
        and receipt.get('deployment_sha256')==deploy_sha and receipt.get('git_commit')==deploy['git_commit']
        and receipt.get('calculation_sha256')==deploy['calculation_sha256'],'K2 machine acceptance identity differs')
    coverage=receipt['pools'][pool]
    require(coverage['pool_bindings']==dict(training=TRAINING[pool],evaluation=EVALUATION),'Wrong acceptance pool')
    for name in required_cases(machine,scope):
        item=coverage['cases'].get(name,{})
        require(item.get('passed') is True,'K2 machine coverage missing: '+name)
        bindings=item.get('config_sha256',{})
        entries=([name] if name in SHORT else ['sgs_default-100','sgs_sharp-100'] if name.startswith('dual')
                 else ['sgs_sharp-20','sgs_sharp-100'])
        expected={case: deploy['files_sha256'][next(job['config'] for job in deploy['verification']
                  if job['pool']==pool and job['machine']==machine and job['case']==case)] for case in entries}
        require(bindings==expected,'K2 acceptance configuration SHA256 differs: '+name)
        require(item.get('evidence') and checked_reference(root,item['evidence'])['passed'],
                'Missing passing evidence for '+name)


def admission_metadata(config, release, gpu, machine, release_sha, acceptance_sha):
    # Deployment permits contain both pools; validated run configuration selects one.
    return dict(machine=machine,gpu=gpu,main_pool=config['main_pool'],
        git_commit=release['git_commit'],release_sha256=release_sha,acceptance_sha256=acceptance_sha,
        protocol_revision=REVISION,scope=release['scope'],arm_id=config['arm_id'],kernel_id=config['kernel_id'])


def admit(root, config_path, config, verify, envs, out, device, machine, release_sha, acceptance_sha, scope):
    require(config.get('protocol_revision')==REVISION and config.get('launch_status')==scope,'Only the K2 protocol is admitted')
    release, deploy=load_release(root,release_sha,scope)
    key=(config['main_pool'],machine,config['arm_id'],config['student_seed'],config['verification_case'])
    entries=deploy['verification'] if scope=='gpu_acceptance' else release['jobs']
    match=next((j for j in entries if key==(j['pool'],j['machine'],j['arm_id'],j['seed'],j['case'])),None)
    require(match is not None and device=='cuda:0' and envs==match['num_envs'],'Wrong K2 arm, machine, seed, device or budget')
    case = match['case']
    require(verify==(match['updates'] if case else None) and
            config==config_for(key[0],key[2],key[3],machine,scope=scope,case=case),'Wrong K2 configuration')
    require(config_path.resolve()==(root/match['config']).resolve(),'Configuration outside frozen release')
    wanted=(root if machine=='yanglab' else root.parent)/'runs'/match['label']
    require(out.absolute().resolve()==wanted.resolve() and not out.exists() and not out.is_symlink() and
            not out.parent.is_symlink(),'Output directory differs or would overwrite a run')
    for b in (TRAINING[key[0]], EVALUATION):
        require(digest(root/b['path'])==b['file_sha256'],'Wrong K2 pool file: '+b['path'])
    gpu=machine_gpu(machine)
    if case:
        require(acceptance_sha is None,'Short validation does not require a long-run receipt')
    else:
        validate_acceptance(root,acceptance_sha,machine,gpu,key[0],scope,deploy,release['deployment']['sha256'])
    return release,gpu
