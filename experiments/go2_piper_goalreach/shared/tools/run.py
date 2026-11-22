"""Bounded simulator checks and a separately guarded, prepared Phase 1 entry point."""
import argparse
import json
import math
from pathlib import Path
import time
import traceback
import random
from isaaclab.app import AppLauncher

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['probe','resets','reference','ppo','benchmark','phase1','p2a','p2b','p3-engineering','p3b','p3s1','p3s2','zero-trace'],required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--pool',type=Path,default=Path('/workspace/pools/train.npz'))
p.add_argument('--evaluation-pool',type=Path,default=Path('/workspace/pools/evaluation.npz'))
p.add_argument('--num-envs',type=int,default=64)
p.add_argument('--reward',choices=['R0','R1'],default='R0')
p.add_argument('--timeout-penalty',type=float,choices=[-100.,-50.],default=-50.)
p.add_argument('--policy',choices=['zero','random','gaussian'],default='zero')
p.add_argument('--noise-sigma',type=float,default=1.)
p.add_argument('--noise-scope',choices=['all','legs','arm'],default='all')
p.add_argument('--start',type=int,default=0)
p.add_argument('--count',type=int,default=252)
p.add_argument('--reset-count',type=int,default=4096)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--allow-phase1',action='store_true')
p.add_argument('--allow-benchmark',action='store_true')
p.add_argument('--phase1-config',type=Path)
p.add_argument('--throttle',type=float,default=.005)
p.add_argument('--diagnostic-capture',action='store_true')
p.add_argument('--p2a-config',type=Path)
p.add_argument('--p2b-config', type=Path)
p.add_argument('--p3-engineering-sha256')
p.add_argument('--p3-yanglab-kernel', choices=('default','sharp'))
p.add_argument('--p3b-deployment-sha256')
p.add_argument('--p3b-scope',choices=('bridge','engineering','formal'))
p.add_argument('--p3b-machine',choices=('yanglab','rental-4060ti','rental-30817','rental-dual3060-26957'))
p.add_argument('--p3b-arm',choices=('table_sharp','neural_sharp','grid16_sharp','grid32_sharp'))
p.add_argument('--p3b-permit-sha256')
p.add_argument('--p3s2-job')
p.add_argument('--p2b-K2-scope', dest='p2b_k2_scope', choices=('gpu_acceptance','provisional_natural_uniform','formal'))
p.add_argument('--p2b-formal-machine', choices=('yanglab', 'rental-4060ti'))
p.add_argument('--p2b-formal-release-sha256')
p.add_argument('--p2b-formal-acceptance-sha256')
p.add_argument('--p2b-verify-updates', type=int, choices=(2, 20, 100, 200))
p.add_argument('--p2a-smoke',action='store_true')
p.add_argument('--p2a-gpu-acceptance', choices=('smoke', 'benchmark', 'rehearsal'))
p.add_argument('--smoke-updates',type=int,choices=[2,4],default=2)
p.add_argument('--smoke-open-curriculum',action='store_true')
p.add_argument('--p2a-capture',action='store_true')
p.add_argument('--legacy-jsonl',action='store_true')
p.add_argument('--initial-action-std',type=float)
p.add_argument('--gamma',type=float)
p.add_argument('--entropy-coef', type=float)
p.add_argument('--success-criterion', choices=('hold_and_speed', 'hold_only'))

AppLauncher.add_app_launcher_args(p)
args=p.parse_args()
if bool(args.p3s2_job)!=(args.mode=='p3s2'):
    p.error('P3S2 requires one explicit declared alias')
phase_protocol=None
p2a_protocol=None
p2b_protocol=None
p2b_formal=None
p3_engineering=None
p3b=None
if args.mode=='p2a':
    import sys
    assert '--reward' in sys.argv,'Reward R0 or R1 must be explicitly supplied'
    assert args.p2a_config is not None
    p2a_protocol=json.loads(args.p2a_config.read_text())
    from tools.p2a_protocol import validate
    validate(p2a_protocol)
    if args.p2a_gpu_acceptance:
        assert not args.p2a_smoke and str(args.device).startswith('cuda')
        assert args.num_envs == (64 if args.p2a_gpu_acceptance == 'smoke' else 4096)
        if args.p2a_gpu_acceptance == 'smoke' and p2a_protocol['updates'] == 8000 and p2a_protocol['arm'] == 'A4':
            assert (args.reward, args.initial_action_std, args.entropy_coef) == ('R0', .3, .001)
        else:
            assert (args.reward, args.initial_action_std, args.entropy_coef) == ('R1', 1., .005)
        assert args.success_criterion == 'hold_only'
        if args.p2a_gpu_acceptance == 'rehearsal':
            assert p2a_protocol['sampler'] == 'manual'
    else:
        assert (args.p2a_smoke and args.device=='cpu' and args.num_envs<=64) or (not args.p2a_smoke and args.num_envs==4096)
    args.p2a_capture=True
    args.seed=p2a_protocol['student_seed']
    project=Path(__file__).resolve().parents[1]
    args.pool=project/p2a_protocol['training_pool']
    args.evaluation_pool=project/p2a_protocol['evaluation_pool']
    args.timeout_penalty=-50.
    import numpy as np
    from tools.build_pools import content_hash
    for path,key in [(args.pool,'training_pool_content_sha256'),(args.evaluation_pool,'evaluation_pool_content_sha256')]:
        assert content_hash(dict(np.load(path,allow_pickle=False)))==p2a_protocol[key]

if args.mode in ('p2b','p3-engineering','p3b','p3s1','p3s2'):
    import sys
    import hashlib
    import numpy as np
    from tools.build_pools import content_hash
    from tools.p2b_protocol import validate, verify_pilot_release
    if args.mode=='p2b' and args.p2b_config is None:
        p.error('--mode p2b requires --p2b-config')
    controlled = ('--reward', '--timeout-penalty', '--seed', '--pool', '--evaluation-pool',
                  '--initial-action-std', '--gamma', '--entropy-coef', '--success-criterion',
                  '--diagnostic-capture', '--smoke-open-curriculum', '--smoke-updates', '--p2a-smoke', '--p2a-config')
    if any(flag.split('=', 1)[0] in controlled for flag in sys.argv[1:]) or args.throttle != 0:
        p.error('2B takes experiment settings from its config and requires --throttle 0')
    project = Path(__file__).resolve().parents[1]
    if args.mode=='p3-engineering':
        if args.p3_yanglab_kernel:
            from tools.p3_yanglab_protocol import admission
        else:
            from tools.p3_engineering_protocol import admission
        if any((args.p2b_config,args.p2b_formal_machine,args.p2b_formal_release_sha256,
                args.p2b_formal_acceptance_sha256,args.p2b_k2_scope,args.p2b_verify_updates,
                args.p2a_gpu_acceptance,args.legacy_jsonl)):
            p.error('P3 engineering uses its own bounded package')
        p3_engineering,p2b_protocol = admission(project,args)
    elif args.mode in ('p3b','p3s1','p3s2'):
        if args.mode=='p3s2':
            from tools.p3s2_protocol import admission
        elif args.mode=='p3s1':
            from tools.p3s1_protocol import admission
        else:
            from tools.p3b_protocol import admission
        if any((args.p2b_config,args.p2b_formal_machine,args.p2b_formal_release_sha256,
                args.p2b_formal_acceptance_sha256,args.p2b_k2_scope,args.p2b_verify_updates,
                args.p2a_gpu_acceptance,args.legacy_jsonl,args.p3_engineering_sha256,args.p3_yanglab_kernel,
                args.allow_phase1,args.allow_benchmark,args.phase1_config)):
            p.error('P3B1 requires its own deployment and scope')
        p3b,p2b_protocol = admission(project,args)
    else:
        p2b_protocol = json.loads(args.p2b_config.read_text())
    if args.p2b_formal_machine:
        from tools.p2b_K2_protocol import admit, admission_metadata, execution as expected_execution
        if args.legacy_jsonl:
            p.error('Formal 2B uses the released binary logging settings')
        release, gpu = admit(project, args.p2b_config, p2b_protocol, args.p2b_verify_updates,
            args.num_envs, args.out, args.device, args.p2b_formal_machine,
            args.p2b_formal_release_sha256, args.p2b_formal_acceptance_sha256, args.p2b_k2_scope)
        p2b_formal = admission_metadata(p2b_protocol,release,gpu,args.p2b_formal_machine,
            args.p2b_formal_release_sha256,args.p2b_formal_acceptance_sha256)
    elif not (p3_engineering or p3b):
        if args.p2b_formal_release_sha256 or args.p2b_formal_acceptance_sha256 or args.p2b_k2_scope:
            p.error('Formal release flags require the formal machine')
        validate(p2b_protocol, args.p2b_verify_updates, args.num_envs, args.out.name, args.device)
        if args.p2b_verify_updates is None:
            verify_pilot_release(project, p2b_protocol)
    args.seed = p2b_protocol['student_seed']
    args.reward = p2b_protocol['reward']
    args.timeout_penalty = p2b_protocol['timeout_penalty']
    args.p2a_capture = True
    pool_bindings = {} if not p3b else {role:p3b['pool'][role] for role in ('training','evaluation')}
    for role in (() if p3b else ('training', 'evaluation')):
        path = project/p2b_protocol[role+'_pool']
        with np.load(path, allow_pickle=False) as data:
            digest = content_hash(dict(data))
            count = p2b_protocol[role+'_pool_size']
            if digest != p2b_protocol[role+'_pool_content_sha256'] or not np.array_equal(
                    data['task_id'], np.arange(count)):
                raise ValueError('2B pool content or task IDs mismatch: ' + str(path))
        pool_bindings[role] = dict(path=p2b_protocol[role+'_pool'], content_sha256=digest,
                                   file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size=count)
    args.pool = project/pool_bindings['training']['path']
    args.evaluation_pool = project/pool_bindings['evaluation']['path']
elif (args.p2b_config is not None or args.p2b_verify_updates is not None or
      args.p2b_k2_scope or args.p2b_formal_machine or args.p2b_formal_release_sha256 or args.p2b_formal_acceptance_sha256):
    p.error('2B flags require --mode p2b')

if any((args.p3b_deployment_sha256,args.p3b_scope,args.p3b_machine,args.p3b_arm,args.p3b_permit_sha256)) and args.mode not in ('p3b','p3s1','p3s2'):
    p.error('P3B1 flags require --mode p3b')

if (args.p3_engineering_sha256 or args.p3_yanglab_kernel) and args.mode!='p3-engineering':
    p.error('P3 engineering package requires its own mode')

if args.p2a_gpu_acceptance and args.mode != 'p2a':
    p.error('--p2a-gpu-acceptance requires --mode p2a')

if args.mode=='phase1':
    assert args.allow_phase1 and args.num_envs==4096
    protocol_path=args.phase1_config or Path(__file__).resolve().parents[1]/'configs'/f'phase1_{args.reward}.json'
    phase_protocol=json.loads(protocol_path.read_text())
    expected=dict(phase=1,reward=args.reward,sampler='uniform',region='near',num_envs=4096,
        updates=1500,rollout_steps=24,from_scratch=True,evaluation_every_updates=100,
        evaluation_pool_size=1008,evaluation_policies=['raw','ema099'],ema_decay=.99)
    assert all(phase_protocol[k]==v for k,v in expected.items()),'Phase 1 protocol differs from the prepared specification'
    args.seed=phase_protocol['student_seed']
    project=Path(__file__).resolve().parents[1]
    args.pool=project/phase_protocol['training_pool']
    args.evaluation_pool=project/phase_protocol['evaluation_pool']
    args.timeout_penalty=phase_protocol['timeout_penalty']
    import numpy as np
    from tools.build_pools import content_hash
    for path,key in [(args.pool,'training_pool_content_sha256'),(args.evaluation_pool,'evaluation_pool_content_sha256')]:
        assert content_hash(dict(np.load(path,allow_pickle=False)))==phase_protocol[key],f'Pool hash mismatch: {path}'
elif args.mode=='benchmark':
    assert args.allow_benchmark and args.num_envs==4096 and args.reward=='R0' and args.timeout_penalty==-50
    assert args.device!='cpu','The 4096-environment measurement is GPU-only'
elif args.mode not in ('p2a', 'p2b', 'p3-engineering','p3b','p3s1','p3s2'):assert 1<=args.num_envs<=64
paired_protocol = p2b_protocol or p2a_protocol
protocol = paired_protocol or phase_protocol or {}
entropy_coef = args.entropy_coef if args.entropy_coef is not None else protocol.get('entropy_coef', .005)
if not math.isfinite(entropy_coef) or entropy_coef < 0:
    p.error('--entropy-coef must be finite and nonnegative')
success_criterion = args.success_criterion or protocol.get('success_criterion', 'hold_and_speed')
if success_criterion not in ('hold_and_speed', 'hold_only'):
    p.error('unknown success criterion')
if args.device=='cpu':
    assert 1<=args.num_envs<=64, 'CPU simulation is limited to 64 environments'
    assert not list(Path('/dev').glob('nvidia*')) and not Path('/dev/dri').exists(), 'CPU acceptance must have no GPU devices'
args.out.mkdir(parents=True,exist_ok=False)
if args.mode=='p3s2':
    from tools.p3s2_protocol import launcher_class
    AppLauncher=launcher_class(AppLauncher,args.p3s2_renderer)
app=AppLauncher(args).app


def main():
    import copy
    import hashlib
    import numpy as np
    import torch
    if args.device=='cpu':
        import warp as wp
        from warp.build import init_kernel_cache
        init_kernel_cache(str(args.out.resolve()/'warp_cache'))
    from goalreach.config import build_cfg,make_env,agent_cfg
    from goalreach.core import write_json,append_rows,REGIONS
    from goalreach.mdp import TaskRuntime
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    def gpu_phase(phase, update=None):
        if args.device != 'cpu':
            marker = args.out/'gpu_phase.json'
            temporary = marker.with_suffix('.tmp')
            write_json(temporary, dict(phase=phase, update=update))
            temporary.replace(marker)

    if args.mode == 'resets':
        gpu_phase('reset_audit')
    pool=args.evaluation_pool if args.mode=='reference' else args.pool
    cfg=build_cfg(pool,args.out,num_envs=args.num_envs,seed=args.seed,reward=args.reward,
        sampler=paired_protocol['sampler'] if paired_protocol else 'evaluation' if args.mode in ('reference','resets') else 'uniform',
        region='near' if args.mode in ('ppo','benchmark','phase1') else None,device=args.device or 'cuda:0',
        timeout_penalty=args.timeout_penalty)
    cfg.goalreach['success_criterion'] = success_criterion
    compare_hold_speed = success_criterion == 'hold_only'
    cfg.goalreach['diagnostic_capture']=args.diagnostic_capture
    cfg.goalreach['training_metrics']=args.mode in ('ppo','benchmark','phase1','p2a','p2b','p3-engineering','p3b','p3s1','p3s2')
    cfg.goalreach.update(p2a_capture=args.p2a_capture,binary_log=args.p2a_capture,
        legacy_jsonl=(args.legacy_jsonl or args.p2a_smoke or args.mode not in ('p2a', 'p2b', 'p3-engineering','p3b','p3s1','p3s2')),
        curriculum=paired_protocol['curriculum'] if paired_protocol else {})
    if p2b_protocol:
        cfg.seed = p2b_protocol['environment_seed']
        cfg.goalreach.update(sampler_seed=p2b_protocol['sampler_seed'],
            p2b_sgs_log=p2b_protocol['sampler'] == 'sgs',
            p2b_trace=(args.p2b_verify_updates is not None or
                       (args.mode=='p3s1' and args.p3b_scope=='bridge')),
            pool_content_sha256=pool_bindings['training']['content_sha256'])
    if p2b_formal:
        cfg.goalreach.update(sgs_kernel_id=p2b_protocol['kernel_id'],
            k2_episode_metadata=dict(protocol_revision='K2', run_id=args.out.name,
                arm_id=p2b_protocol['arm_id'], kernel_id=p2b_protocol['kernel_id'],
                sgs=p2b_protocol['sgs'], training_pool=p2b_protocol['training_pool'],
                episode_semantics='Natural endings only; evaluation resets are recorded separately'))
    if p3_engineering:
        cfg.goalreach.update(p3_neural=p3_engineering['staging_options_awaiting_2c'],
            p3_yanglab_telemetry=bool(args.p3_yanglab_kernel),
            k2_episode_metadata=dict(stage='3-engineering',protocol_revision='K2',
                run_id=args.out.name,arm_id='neural_engineering',kernel_id=p2b_protocol['kernel_id']))
    if p3b:
        job=p3b.get('job',{})
        legacy_bridge=job.get('kind')=='bridge_old'
        cfg.goalreach.update(p3b=not legacy_bridge,sgs_kernel_id='sharp' if legacy_bridge else None,p2b_sgs_log=True,
            p2b_trace=cfg.goalreach.get('p2b_trace') or job.get('kind') in ('bridge_old','bridge_new','parallel','accept'),
            p3b_feature_binding=p3b['feature_binding'],p3s1_grid_bins=p3b.get('grid_bins'),
            p3_neural=p3b['scorer_settings'] if args.p3b_arm=='neural_sharp' else None,
            k2_episode_metadata=dict(stage={'p3s2':'P3S2','p3s1':'P3S1'}.get(args.mode,'P3B1'),scope=args.p3b_scope,run_id=args.out.name,
                arm_id=args.p3b_arm,kernel_id='sharp',sgs=p2b_protocol['sgs']))
    cfg.goalreach['training_update']=1
    if args.smoke_open_curriculum:
        assert args.p2a_smoke and args.device=='cpu'
        cfg.goalreach['curriculum']=dict(success_threshold=0.,minimum_episodes=1)
    if cfg.goalreach.get('p2b_trace'):
        from tools.p2b_instrument import trace_assignments
        assignment_trace = trace_assignments(TaskRuntime)
    start=time.monotonic()
    env=make_env(cfg)
    try:
        initial_std=args.initial_action_std if args.initial_action_std is not None else protocol.get('initial_action_std',1.)
        gamma=args.gamma if args.gamma is not None else (paired_protocol or {}).get('gamma',.99)
        assert initial_std>0 and 0<gamma<=1
        agent = agent_cfg(args.seed, initial_std, entropy_coef)
        agent.algorithm.gamma=gamma

        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        if str(env.device).startswith('cuda') and args.mode in ('ppo', 'benchmark', 'phase1', 'p2a', 'p2b', 'p3-engineering','p3b','p3s1','p3s2'):
            # Use the GPU environment boundary validated in Dialogue 7.
            for method in ('reset', 'step', 'get_observations'):
                setattr(wrapped, method, torch.inference_mode()(getattr(wrapped, method)))
        rt=env.goal_runtime
        zero=torch.zeros(env.num_envs,18,device=env.device)
        obs=wrapped.get_observations()
        assert obs['policy'].shape==(env.num_envs,201),obs['policy'].shape
        assert not env.command_manager.active_terms,env.command_manager.active_terms
        assert env.max_episode_length==500 and env.cfg.is_finite_horizon
        def step(action):
            result=wrapped.step(action)
            for v in result[0].values():assert torch.isfinite(v).all()
            assert torch.isfinite(result[1]).all()
            assert 'time_outs' not in result[3] or not result[3]['time_outs'].any()
            if args.throttle:time.sleep(args.throttle)
            return result
        def force_tasks(ids):
            rt.forced_ids=list(map(int,ids))
            wrapped.reset()
            return wrapped.get_observations()
        def policy_evaluation(policy, task_ids):
            rows=[]
            for k in range(0,len(task_ids),env.num_envs):
                batch=task_ids[k:k+env.num_envs];valid=len(batch)
                padded=list(batch)+[int(batch[-1])]*(env.num_envs-valid)
                ob=force_tasks(padded)
                generations=rt.state.generation.clone()
                seen=set();rt.last_rows=[]
                for _ in range(500):
                    with torch.inference_mode():a=policy(ob)
                    ob,_,_,_=step(a)
                    for row in rt.last_rows:
                        i=row['env_id']
                        if i<valid and i not in seen and row['generation']==int(generations[i]):
                            rows.append(row);seen.add(i)
                    rt.last_rows=[]
                    if len(seen)==valid:break
                assert len(seen)==valid,(len(seen),valid)
            return rows
        def summarize(rows):
            return {r:dict(n=sum(x['region']==i for x in rows),
                          successes=sum(x['region']==i and x['success'] for x in rows),
                          success_rate=(sum(x['region']==i and x['success'] for x in rows)/sum(x['region']==i for x in rows)) if any(x['region']==i for x in rows) else None)
                    for i,r in enumerate(REGIONS)}

        def near_bands(rows):
            return {label:dict(n=sum(x.get('near_distance_band')==i for x in rows),
                successes=sum(x.get('near_distance_band')==i and x['success'] for x in rows),
                success_rate=(sum(x.get('near_distance_band')==i and x['success'] for x in rows)/sum(x.get('near_distance_band')==i for x in rows)) if any(x.get('near_distance_band')==i for x in rows) else None)
                for i,label in enumerate(('0-2cm','2-5cm','5-10cm'))}

        result=dict(mode=args.mode,num_envs=env.num_envs,reward=args.reward,actor_dim=201,
            timeout_penalty=args.timeout_penalty,
            execution_settings=dict(reward=args.reward,initial_action_std=initial_std,gamma=gamma,
                entropy_coef=agent.algorithm.entropy_coef,success_criterion=success_criterion,student_seed=args.seed),
            simulation_device=str(env.device),
            critic_dim=obs['critic'].shape[-1],action_dim=18,step_dt=env.step_dt,
            timeout_is_termination=True,command_terms=env.command_manager.active_terms,
            reward_weights={n:env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager.active_terms})
        if p2b_protocol:
            from goalreach.core import PARAMETERS
            import importlib.metadata as metadata
            import platform
            actual_sgs = rt.sampler.kernel_settings() if rt.sampler.mode == 'sgs' else None
            if p3b:
                assert rt.sampler.kernel_settings()==p2b_protocol['sgs']
                actual_sgs=p2b_protocol['sgs']
            elif not p3_engineering:
                assert p2b_protocol['sgs'] == (actual_sgs if p2b_formal else PARAMETERS)
            else:
                from goalreach.sgs_kernels import KERNELS
                actual_sgs = dict(KERNELS[rt.neural.estimator.options.kernel])
                assert actual_sgs==p2b_protocol['sgs']
                result['engineering_only'] = True
            result['execution_settings'].update(environment_seed=int(env.cfg.seed),
                sampler_seed=rt.sampler_seed, sampler=p2b_protocol['sampler'],
                rollout_steps=24, ema_decay=.99, success_reward=1000., physical_failure_reward=-100.,
                timeout_penalty=args.timeout_penalty, success_radius_m=.05, hold_steps=25,
                timeout_steps=500, episode_seconds=10., pools=pool_bindings,
                sgs=actual_sgs,
                evaluation_every_updates=100, raw_evaluation_every_updates=500,
                checkpoint_every_updates=500, configured_updates=p2b_protocol['updates'],
                executed_updates=args.p2b_verify_updates or p2b_protocol['updates'])
            assert result['execution_settings']['environment_seed'] == p2b_protocol['environment_seed']
            assert result['execution_settings']['sampler_seed'] == p2b_protocol['sampler_seed']
            result['versions'] = dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, cuda=torch.version.cuda,
                **{name:metadata.version(name) for name in ('rsl-rl-lib','isaaclab','isaacsim')})
            if p3b:
                from tools.p3_yanglab_protocol import active_uuid
                assert active_uuid()==p3b['host']['uuid']
                result['p3b_admission']=dict(machine=args.p3b_machine,scope=args.p3b_scope,arm=args.p3b_arm,
                    gpu_uuid=p3b['host']['uuid'],deployment_sha256=args.p3b_deployment_sha256,
                    permit_sha256=args.p3b_permit_sha256,selection_sha256=p3b['selection_sha256'])
                if args.mode=='p3s2':
                    result['p3b_admission'].update(job=args.p3s2_job,protocol_sha256=p3b['protocol_sha256'])
                write_json(args.out/'p3b_admission.json',result['p3b_admission'])
                result['execution_settings'].update(checkpoint_every_updates=p2b_protocol['checkpoint_every_updates'],
                    raw_evaluation_every_updates=p2b_protocol['raw_evaluation_every_updates'])
            if p3_engineering and result['versions']!=p3_engineering['simulator_dependencies']:
                raise ValueError('P3 simulator dependency versions differ')
            if p3_engineering:
                result['execution_settings'].update(evaluation_every_updates=20,
                    raw_evaluation_every_updates=20,checkpoint_every_updates=20)
            if args.p3_yanglab_kernel:
                from tools.p3_yanglab_protocol import GPU,active_uuid
                assert active_uuid()==GPU
                result['engineering_admission']=dict(scope='engineering_only',
                    package_sha256=args.p3_engineering_sha256,kernel=args.p3_yanglab_kernel,
                    physical_gpu_uuid=GPU)
                write_json(args.out/'engineering_admission.json',result['engineering_admission'])
            if p2b_formal:
                result['execution_settings'].update(protocol_revision='K2',
                    arm_id=p2b_protocol['arm_id'], kernel_id=rt.sampler.kernel_id,
                    empty_history_prior=.5 if actual_sgs else None)
                if result['execution_settings'] != expected_execution(
                        p2b_protocol, args.p2b_verify_updates or 8000):
                    raise ValueError('Actual training settings differ from formal release')
                if result['versions'] != release['dependency_versions']:
                    raise ValueError('Runtime dependencies differ from formal release')
                result['formal_admission'] = p2b_formal
                write_json(args.out/'formal_admission.json', p2b_formal)
            write_json(args.out/'p2b_protocol.json',p2b_protocol)
            write_json(args.out/'execution_settings.json',result['execution_settings'])
        if phase_protocol is not None:
            result['phase1_protocol']=phase_protocol
            write_json(args.out/'phase1_protocol.json',phase_protocol)
        if str(env.device).startswith('cuda'):
            result.update(cuda_device=torch.cuda.get_device_name(),
                cuda_environment_allocated_mib=torch.cuda.memory_allocated()/2**20,
                cuda_environment_reserved_mib=torch.cuda.memory_reserved()/2**20,
                physx_capacity={k:v for k,v in cfg.sim.physx.to_dict().items() if k.startswith('gpu_')})
        if args.mode=='zero-trace':
            records=[]
            for j in range(501):
                r=env.scene['robot'].data
                if j%10==0:
                    records.append(dict(step=j,ee=(r.body_pos_w[:,rt.ee]-env.scene.env_origins).cpu().tolist(),
                        expected_initial_ee=rt.initial_ee[rt.state.task].cpu().tolist(),
                        ee_speed=r.body_link_lin_vel_w[:,rt.ee].norm(dim=-1).cpu().tolist(),
                        root=(r.root_pos_w-env.scene.env_origins).cpu().tolist(),
                        hold=rt.state.hold.cpu().tolist(),generation=rt.state.generation.cpu().tolist()))
                if j<500:step(zero)
            write_json(args.out/'trace.json',records)
            result.update(initial=records[0],at_half_second=records[3],at_one_second=records[5],at_five_seconds=records[25])
        elif args.mode=='probe':
            initial=env.scene['robot'].data.root_pos_w.clone();checked=0
            result['end_effector_com_offset_m']=env.scene['robot'].data.body_com_pos_b[:,rt.ee].cpu().tolist()
            measured_start=None
            for j in range(100):
                if j==20:
                    if str(env.device).startswith('cuda'):torch.cuda.synchronize()
                    measured_start=time.monotonic()
                targets=rt.state.target.clone();generation=rt.state.generation.clone()
                step(torch.rand_like(zero)*2-1)
                same=generation==rt.state.generation
                assert torch.equal(targets[same],rt.state.target[same])
                checked+=int(same.sum())
            if str(env.device).startswith('cuda'):torch.cuda.synchronize()
            result.update(measured_control_steps=80,
                measured_step_wall_s=(time.monotonic()-measured_start)/80,
                deliberate_sleep_per_step_s=args.throttle,
                timing_includes='random actions, physics, observations, rewards, terminal feedback and assertions')
            result['world_invariance_transitions']=checked
            result['base_displacement_max_m']=float((env.scene['robot'].data.root_pos_w-initial).norm(dim=-1).max())
            # Inject the measured trajectory, not task state, to verify the live
            # termination manager -> reward -> autoreset -> feedback integration.
            force_tasks(range(env.num_envs));generation=rt.state.generation.clone()
            native_tick=rt.tick
            def constructed_tick():
                if rt.last_tick==env.common_step_counter:return
                rt.last_tick=env.common_step_counter
                no=torch.zeros(env.num_envs,device=env.device,dtype=torch.bool)
                rt.state.advance(rt.state.target.clone(),torch.zeros_like(rt.state.target),no,no,no)
            rt.tick=constructed_tick;rt.last_rows=[]
            for i in range(25):
                _,rewards,dones,_=step(zero)
                assert bool(dones.all())==(i==24)
            assert (rt.state.generation==generation+1).all()
            assert len(rt.last_rows)==env.num_envs
            assert all(r['success'] and r['episode_steps']==25 and r['terminal_distance_m']==0 for r in rt.last_rows)
            result['constructed_25_step_live_autoreset']=True
            result['success_step_reward_min']=float(rewards.min())
            _,rewards,dones,_=step(zero)
            assert not dones.any() and not rt.last_rows and float(rewards.max()) < 100.
            assert (rt.state.generation==generation+1).all()
            result['success_reward_not_repeated_after_reset']=True
            rt.tick=native_tick
        elif args.mode=='resets':
            from tools.collision_audit import CollisionAudit
            collision=CollisionAudit(env)
            write_json(args.out/'collision_assets.json',dict(hashes=collision.hashes,
                collision_prims=[{k:v for k,v in x.items() if k!='corners'} for x in collision.entries]))
            all_ids=np.linspace(0,len(rt.targets)-1,args.reset_count,dtype=int)
            assert len(np.unique(all_ids))==args.reset_count and args.reset_count%env.num_envs==0
            initial_fail=penetrations=initial_success=nonnear_inside=0;min_foot=float('inf');max_force=0.;fk_error=0.
            footprint=env.scene['robot'].find_bodies('.*_foot')[0]
            # USD foot spheres are verified in the separate asset audit.
            reset_rows=[]
            for k in range(0,len(all_ids),env.num_envs):
                ids=all_ids[k:k+env.num_envs];force_tasks(ids)
                env.sim.forward();env.scene.update(env.physics_dt)
                robot=env.scene['robot'].data
                z=collision.clearance(env)
                min_foot=min(min_foot,float(z.min()));penetrations+=int((z < -.002).sum())
                d=(rt.state.target-robot.body_pos_w[:,rt.ee]).norm(dim=-1)
                expected=rt.initial_ee[rt.state.task]+env.scene.env_origins
                fk_error=max(fk_error,float((expected-robot.body_pos_w[:,rt.ee]).norm(dim=-1).max()))
                reg=torch.as_tensor(rt.pool['region'][ids],device=env.device)
                nonnear_inside+=int(((d<=.05)&(reg!=0)).sum())
                initial_success+=int((rt.state.hold>=25).sum())
                generation=rt.state.generation.clone();rt.last_rows=[]
                initial=[]
                for _ in range(26):
                    step(zero)
                    initial.extend(r for r in rt.last_rows if r['generation']==int(generation[r['env_id']]))
                initial_fail+=sum(not r['success'] for r in initial)
                reset_rows.extend(initial)
                max_force=max(max_force,float(env.scene.sensors['contact_forces'].data.net_forces_w_history[:,:,rt.nonfoot].norm(dim=-1).max()))
            result.update(configurations=args.reset_count,initial_penetration_count=penetrations,
                min_initial_collision_aabb_clearance_m=min_foot,early_failure_count=initial_fail,
                initial_success_count=initial_success,first_26_step_terminals=summarize(reset_rows),
                first_26_step_near_bands=near_bands(reset_rows),
                max_observed_nonfoot_force_n=max_force)
            result.update(initial_nonnear_within_radius=nonnear_inside,max_fk_simulation_error_m=fk_error)
            result['passed']=penetrations==initial_fail==initial_success==nonnear_inside==0 and fk_error<1e-4
        elif args.mode=='reference':
            policy=(lambda ob:zero) if args.policy=='zero' else (lambda ob:torch.rand_like(zero)*2-1)
            from goalreach.diagnostics import gaussian_action,episode_summary
            if args.policy=='gaussian':policy=lambda ob:gaussian_action(zero,args.noise_sigma,args.noise_scope)
            ids=list(range(args.start,min(args.start+args.count,len(rt.targets))))
            rows=policy_evaluation(policy,ids)
            write_json(args.out/'evaluation_rows.json',rows)
            result.update(policy=args.policy,range=[args.start,args.start+len(ids)],regions=summarize(rows),near_distance_bands=near_bands(rows))
            result.update(episode_summary=episode_summary(rows, compare_hold_speed),noise_sigma=args.noise_sigma if args.policy=='gaussian' else None,
                noise_scope=args.noise_scope if args.policy=='gaussian' else None,action_clip=agent.clip_actions)
        else:
            class Runner(OnPolicyRunner):
                def save(self,path,infos=None):
                    if args.mode=='phase1':return super().save(path,infos)
            runner=Runner(wrapped,agent.to_dict(),log_dir=str(args.out/'policy'),device=env.device)
            actor=runner.alg.actor
            from goalreach.training_metrics import TrainingWindow
            from goalreach.diagnostics import episode_summary,hold_speed_comparison
            def action_noise():
                std=actor.distribution.log_std_param.detach().exp().cpu()
                assert std.shape==(18,) and torch.isfinite(std).all()
                return dict(legs_mean=float(std[:12].mean()),arm_mean=float(std[12:].mean()),per_joint=std.tolist())
            initial_noise=action_noise()
            assert np.allclose(initial_noise['per_joint'],initial_std)
            assert agent.algorithm.gamma==gamma
            assert runner.alg.entropy_coef == entropy_coef
            cfg.goalreach['sampling_action_std']=initial_noise
            initial_hash=hashlib.sha256()
            for key,value in sorted(actor.state_dict().items()):
                initial_hash.update(key.encode());initial_hash.update(value.detach().cpu().numpy().tobytes())
            write_json(args.out/'initial_training_state.json',dict(action_std=initial_noise,gamma=gamma,
                entropy_coef=agent.algorithm.entropy_coef,success_criterion=success_criterion,
                student_seed=args.seed,timeout_penalty=args.timeout_penalty,agent=agent.to_dict(),
                actor_content_sha256=initial_hash.hexdigest()))
            if p3b:
                if args.mode=='p3s2':
                    from tools.p3s2_protocol import verify_initial
                elif args.mode=='p3s1':
                    from tools.p3s1_protocol import verify_initial
                else:
                    from tools.p3b_protocol import verify_initial
                verify_initial(p3b,p2b_protocol,initial_hash.hexdigest(),args.out)
            trace = cfg.goalreach.get('p2b_trace')
            if trace:
                from tools.p2b_instrument import write_initial
                if args.mode in ('p3s1','p3s2') and rt.forecast is not None:
                    from tools.p3b_protocol import training_state_digest
                    write_initial(args.out, env, rt, runner, obs, sampler_digest=training_state_digest)
                else:
                    write_initial(args.out, env, rt, runner, obs)
            if p2b_formal and args.p2b_verify_updates is None and p2b_formal['machine'] == 'yanglab':
                from tools.p2b_formal_protocol import read
                queue_path = project/'runtime/p2b_K2_yanglab_status.json'
                queue = read(queue_path)
                current_arm = p2b_protocol['arm_id']
                arm_order = ('uniform', 'manual', 'sgs_default', 'sgs_sharp')
                for previous_arm in arm_order[:arm_order.index(current_arm)]:
                    previous = queue['arms'][previous_arm]
                    if (previous['phase'] != 'complete' or
                            previous['initial']['actor_content_sha256'] != initial_hash.hexdigest()):
                        raise ValueError('Formal seed 43 weights differ before the first training update')
            ema={k:v.detach().clone() for k,v in actor.state_dict().items()}
            if args.mode=='phase1':
                torch.save(dict(actor_state_dict={k:v.detach().clone() for k,v in actor.state_dict().items()},
                    ema_state_dict=ema,completed_updates=0),args.out/'actors_0000.pt')
            updates=[];native_update=runner.alg.update;completed=0
            optimizer_calls=0;native_optimizer_step=runner.alg.optimizer.step
            def record_ppo_budget(started, finished):
                temporary=args.out/'ppo_budget.tmp'
                write_json(temporary,dict(ppo_updates_started=started,ppo_updates_completed=finished,
                                          optimizer_calls=optimizer_calls))
                temporary.replace(args.out/'ppo_budget.json')
            if args.p3_yanglab_kernel or p3b:
                record_ppo_budget(0,0)
            def optimizer_step(*a,**kw):
                nonlocal optimizer_calls
                optimizer_calls+=1
                return native_optimizer_step(*a,**kw)
            runner.alg.optimizer.step=optimizer_step
            def audited_update():
                nonlocal completed
                assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
                if args.p3_yanglab_kernel or p3b:
                    record_ppo_budget(completed+1,completed)
                begin=time.monotonic();losses=native_update();completed+=1
                if args.p3_yanglab_kernel or p3b:
                    record_ppo_budget(completed,completed)
                assert optimizer_calls==completed*20
                assert all(np.isfinite(float(v)) for v in losses.values())
                assert all(torch.isfinite(v).all() for v in [*actor.parameters(),*runner.alg.critic.parameters()])
                with torch.no_grad():
                    for k,v in actor.state_dict().items():
                        if v.is_floating_point():ema[k].mul_(.99).add_(v.detach(),alpha=.01)
                        else:ema[k].copy_(v)
                row=dict(update=completed,optimizer_s=time.monotonic()-begin,losses={k:float(v) for k,v in losses.items()})
                row.update(action_std=action_noise(),learning_rate=float(runner.alg.learning_rate))
                cfg.goalreach.update(training_update=completed+1,sampling_action_std=row['action_std'])
                if args.p2a_capture:rt.end_update(completed)
                updates.append(row);return losses
            runner.alg.update=audited_update
            def save_paired_checkpoint(stop):
                scorer = {'neural':rt.neural.state_dict()} if rt.neural else {}
                if p3b and rt.forecast is not None:
                    scorer['forecast'] = rt.forecast.state_dict()
                torch.save(dict(actor=actor.state_dict(),critic=runner.alg.critic.state_dict(),
                    optimizer=runner.alg.optimizer.state_dict(),ema=ema,sampler=rt.sampler.state_dict(),
                    curriculum=rt.curriculum.state_dict() if rt.curriculum else None,
                    completed_updates=completed,learning_rate=runner.alg.learning_rate,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(),python_rng=random.getstate(),protocol=paired_protocol,
                    agent_config=agent.to_dict(),execution_settings=result['execution_settings'],**scorer),
                    args.out/f'checkpoint_{stop:04d}.pt')

            native_log=runner.logger.log
            def measured_log(*a,**kw):
                if updates:
                    updates[-1].update(rollout_s=float(kw['collect_time']),learn_s=float(kw['learn_time']))
                    updates[-1]['update_wall_s']=float(kw['collect_time'])+float(kw['learn_time'])
                    append_rows(args.out/'updates.jsonl',[updates[-1]])
                    if completed%50==0 or args.mode in ('ppo','benchmark'):
                        row=dict(update=completed,window_completed_episodes_only=True,
                            action_std=updates[-1]['action_std'],value_loss=updates[-1]['losses']['value'],
                            learning_rate=updates[-1]['learning_rate'],**rt.training_window.export())
                        append_rows(args.out/'training_metrics.jsonl',[row])
                        rt.training_window=TrainingWindow()
                    write_json(args.out/'progress.json',dict(completed_updates=completed,
                        action_std=updates[-1]['action_std'],losses=updates[-1]['losses'],
                        learning_rate=updates[-1]['learning_rate'],unix=time.time()))
                value = native_log(*a,**kw)
                if args.mode=='p3s2' and completed==20 and p3b['job']['kind'] in ('accept','parallel'):
                    save_paired_checkpoint(completed)
                    write_json(args.out/'validation_assignments_0020.json',assignment_trace())
                if args.p2b_verify_updates == 100 and completed == 20:
                    save_paired_checkpoint(completed)
                    write_json(args.out/'validation_assignments_0020.json', assignment_trace())
                    rt.episode_log.flush(completed, final=True)
                return value
            runner.logger.log=measured_log
            @torch.inference_mode()
            def evaluate_pair(stop,task_ids,folder,policy_labels=None):
                nonlocal rt
                gpu_phase('evaluation', stop)
                if str(env.device).startswith('cuda'):torch.cuda.synchronize()
                pair_begin=time.monotonic();policy_times={}
                raw={k:v.detach().clone() for k,v in actor.state_dict().items()}
                training_rt=rt;training_cfg=copy.deepcopy(env.cfg.goalreach)
                before=training_rt.sampler.completed_counts.copy();before_updates=completed
                if p2b_protocol:
                    from tools.p2b_instrument import training_state_digest
                    if p3b:
                        from tools.p3b_protocol import training_state_digest
                    before_training_state = training_state_digest(training_rt)
                if args.p3_yanglab_kernel:
                    scorer_before=training_rt.neural.digest()
                rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                numpy_rng=np.random.get_state();python_rng=random.getstate()
                summaries={}
                if policy_labels is None:
                    policy_labels = ('raw', 'ema099') if not paired_protocol or args.p2a_smoke or stop%500==0 else ('ema099',)
                states = {'raw': raw, 'ema099': ema}
                for label in policy_labels:
                    state = states[label]
                    gpu_phase('evaluation_'+label, stop)
                    # Same reset seed. Subsequent noise draws may diverge after
                    # policy-dependent autoresets, as in the native environment.
                    random.seed(2026111209);np.random.seed(2026111209);torch.manual_seed(2026111209)
                    env.cfg.goalreach.update(pool=str(args.evaluation_pool),sampler='evaluation',region=None,
                        training_metrics=False,p2b_trace=False,p2b_sgs_log=False,sgs_kernel_id=None,k2_episode_metadata=None,out=str(args.out/folder/f'{stop:04d}_{label}'))
                    rt=TaskRuntime(env);env.goal_runtime=rt
                    if args.p3_yanglab_kernel or p3b:
                        assert rt.neural is None and rt.curriculum is None and rt.episode_log is None
                    actor.load_state_dict(state);policy=runner.get_inference_policy(device=env.device)
                    if str(env.device).startswith('cuda'):torch.cuda.synchronize()
                    policy_begin=time.monotonic()
                    rows=policy_evaluation(policy,task_ids)
                    if args.p3_yanglab_kernel or p3b:
                        assert not any(rt.sampler.completed_counts)
                    if str(env.device).startswith('cuda'):torch.cuda.synchronize()
                    policy_times[label]=time.monotonic()-policy_begin
                    summaries[label]=summarize(rows)
                    write_json(rt.out/'rows.json',rows);write_json(rt.out/'summary.json',summaries[label])
                    write_json(rt.out/'near_distance_bands.json',near_bands(rows))
                    detail=dict(near=episode_summary([r for r in rows if r['region']==0], compare_hold_speed),
                        regions={name:episode_summary([r for r in rows if r['region']==i], compare_hold_speed) for i,name in enumerate(REGIONS)})
                    if compare_hold_speed:
                        detail.update(hold_speed_comparison(rows))
                    if p3b:
                        for name, sector in (('right',2),('back',3)):
                            detail[name]={f"{region}:{band}":episode_summary([r for r in rows if
                                r['region']==region and r['azimuth_sector']==sector and
                                (band==-1 or (rt.pool['horizontal_distance'][r['task_id']]>=2.25)==band)],compare_hold_speed)
                                for region,band in ((2,-1),(3,0),(3,1))}
                    write_json(rt.out/'metrics.json',detail)
                    append_rows(args.out/'evaluation_curve.jsonl',[dict(update=stop,policy=label,wall_s=policy_times[label],**detail)])
                gpu_phase('evaluation_restore', stop)
                actor.load_state_dict(raw);env.goal_runtime=training_rt;rt=training_rt;env.cfg.goalreach=training_cfg
                assert np.array_equal(before,rt.sampler.completed_counts) and completed==before_updates
                if p2b_protocol:
                    assert before_training_state == training_state_digest(rt)
                assert all(torch.equal(v,actor.state_dict()[k]) for k,v in raw.items())
                discarded={name:dict(count=int(((rt.state.task>=0)&torch.as_tensor(rt.pool['region'][rt.state.task.cpu().numpy().clip(0)]==i,device=env.device)).sum()),
                    elapsed_steps=rt.state.steps[(rt.state.task>=0)&torch.as_tensor(rt.pool['region'][rt.state.task.cpu().numpy().clip(0)]==i,device=env.device)].cpu().tolist()) for i,name in enumerate(REGIONS)}
                append_rows(args.out/'evaluation_censoring.jsonl',[dict(update=stop,regions=discarded,
                    unfinished_training_episodes=int((rt.state.task>=0).sum()),
                    reason='Evaluation boundary; discard incomplete episodes without sampler feedback')])
                torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
                np.random.set_state(numpy_rng);random.setstate(python_rng)
                if p2b_protocol:
                    assert torch.equal(rng, torch.get_rng_state())
                    assert all(torch.equal(a, b) for a,b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
                    assert random.getstate() == python_rng
                    numpy_restored = np.random.get_state()
                    assert np.array_equal(numpy_restored[1], numpy_rng[1])
                    assert numpy_restored[0] == numpy_rng[0] and numpy_restored[2:] == numpy_rng[2:]
                wrapped.reset()
                if p2b_protocol:
                    assert np.array_equal(before,rt.sampler.completed_counts)
                runner.logger.cur_reward_sum.zero_();runner.logger.cur_episode_length.zero_()
                runner.logger.ep_extras=[]
                runner.alg.train_mode()
                audit=dict(raw_weights_restored=True,training_feedback_unchanged=True,
                    updates_unchanged=True,discarded_logger_partials=True,unique_evaluation_tasks=len(task_ids),summaries=summaries,
                    policy_wall_s=policy_times,pair_wall_s=time.monotonic()-pair_begin)
                if p2b_protocol:
                    audit.update(training_sampler_and_curriculum_isolated=True,
                        global_rng_restored_before_training_reset=True,
                        discarded_episodes_did_not_update_table=True)
                if args.p3_yanglab_kernel:
                    assert training_rt.neural.digest()==scorer_before
                    audit['neural_scorer_isolated_through_training_reset']=True
                if p3b:
                    audit['evaluation_feedback_free']=True
                    audit['evaluation_forecast_and_neural_state_isolated']=True
                write_json(args.out/folder/f'{stop:04d}_switch_audit.json',audit)
                return audit
            if args.mode=='ppo':
                if str(env.device).startswith('cuda'):torch.cuda.synchronize()
                begin=time.monotonic();runner.learn(num_learning_iterations=2,init_at_random_ep_len=False)
                if str(env.device).startswith('cuda'):torch.cuda.synchronize()
                elapsed=time.monotonic()-begin;assert completed==2
                torch.save(dict(actor=actor.state_dict(),critic=runner.alg.critic.state_dict(),ema=ema),args.out/'smoke_weights.pt')
                if rt.episode_log:rt.episode_log.flush(completed,final=True)
                storage=runner.alg.storage
                buffers=[*storage.observations.values(),storage.actions,storage.rewards,
                    storage.values,storage.actions_log_prob,storage.returns,storage.advantages,
                    *storage.distribution_params]
                assert all(torch.isfinite(value).all() for value in buffers)
                result['rollout_storage_mib']=(sum(v.numel()*v.element_size() for v in buffers)
                    +storage.dones.numel()*storage.dones.element_size())/2**20
                result['network_parameters']=dict(actor=sum(v.numel() for v in actor.parameters()),
                    critic=sum(v.numel() for v in runner.alg.critic.parameters()))
                # Complete current episodes with the just-created policy to measure
                # actual episode reward sums, without further optimizer updates.
                cfg.goalreach['training_metrics']=False # Deterministic drain is not exploratory training.
                policy=runner.get_inference_policy(device=env.device);ob=wrapped.get_observations()
                initial_generation=rt.state.generation.clone();rt.last_rows=[];first={}
                for _ in range(500):
                    with torch.inference_mode():a=policy(ob)
                    ob,_,_,_=step(a)
                    for row in rt.last_rows:
                        i=row['env_id']
                        if row['generation']==int(initial_generation[i]):first[i]=row
                    rt.last_rows=[]
                    if len(first)==env.num_envs:break
                result.update(updates=updates,learn_wall_s=elapsed,seconds_per_update=elapsed/2,
                    completed_episodes=rt.reward_episodes,
                    reward_episode_means={k:v/max(1,rt.reward_episodes) for k,v in rt.reward_sums.items()},
                    cuda_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                    cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20,
                    finite=True,ema_updates=completed,ema_decay=.99,optimizer_calls=optimizer_calls,
                    training_transitions=completed*24*env.num_envs)
                result['evaluation_switch_smoke']=evaluate_pair(2,[r*252+z*84 for r in range(4) for z in range(3)],'evaluation-smoke')
            elif args.mode=='benchmark':
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
                begin=time.monotonic();runner.learn(num_learning_iterations=20,init_at_random_ep_len=False)
                torch.cuda.synchronize();elapsed=time.monotonic()-begin
                assert completed==20
                result.update(completed_updates=completed,learn_wall_s=elapsed,seconds_per_update=elapsed/20,
                    per_update_wall_s=[r['update_wall_s'] for r in updates],finite=True,ema_updates=completed,
                    training_cuda_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                    training_cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20)
                result['full_evaluation']=evaluate_pair(20,list(range(1008)),'evaluations')
                result.update(cuda_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                    cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20)
            elif paired_protocol:
                acceptance = args.p2a_gpu_acceptance
                if p2b_protocol and args.p2b_verify_updates is not None:
                    acceptance = {2:'smoke',20:'benchmark',100:'rehearsal',200:'rehearsal'}[args.p2b_verify_updates]
                short_smoke = args.p2a_smoke or acceptance == 'smoke'
                smoke = args.mode=='p3s2' and p3b['job']['kind']=='smoke'
                bridge = args.mode in ('p3s1','p3s2') and args.p3b_scope=='bridge'
                if smoke:
                    total = interval = 2
                elif bridge:
                    total = interval = 20
                elif p3_engineering:
                    total, interval = paired_protocol['updates'], 20
                elif short_smoke:
                    total = args.smoke_updates
                    interval = 2
                elif acceptance == 'benchmark':
                    total = interval = 20
                elif acceptance == 'rehearsal':
                    total, interval = args.p2b_verify_updates or 200, 100
                else:
                    total, interval = paired_protocol['updates'], 100

                with np.load(args.evaluation_pool, allow_pickle=False) as data:
                    task_ids = ([int(np.flatnonzero(data['cell_id']==i)[0]) for i in range(42)]
                                if short_smoke or p3_engineering else list(range(len(data['task_id']))))
                for stop in range(interval,total+1,interval):
                    gpu_phase('training', completed)
                    runner.current_learning_iteration=completed
                    runner.learn(num_learning_iterations=interval,init_at_random_ep_len=False)
                    assert completed==stop
                    if rt.episode_log:
                        rt.episode_log.flush(stop,final=bridge or smoke or bool(p3_engineering) or short_smoke or acceptance == 'benchmark')
                    if (p3_engineering or (p3b and stop==total) or stop%500==0 or short_smoke or (acceptance == 'rehearsal' and stop == total)
                            or (trace and acceptance == 'benchmark')):
                        save_paired_checkpoint(stop)
                    if trace:
                        write_json(args.out/'validation_assignments.json', assignment_trace())
                    if bridge or smoke:
                        continue  # The checkpoint is taken before any evaluation.
                    policy_labels = ('raw', 'ema099') if p3_engineering or short_smoke or acceptance == 'benchmark' else None
                    if p3b:
                        raw_due=stop%500==0 or stop==total
                        if args.mode=='p3s2' and p3b['job']['kind']=='parallel':
                            raw_due=False
                        policy_labels=('raw','ema099') if raw_due else ('ema099',)
                    evaluate_pair(stop, task_ids, 'evaluations', policy_labels)
                if short_smoke or acceptance == 'benchmark':
                    append_rows(args.out/'sampling_diagnostics.jsonl',[rt.sampling_diagnostics.export(rt.sampler,completed)])
                result.update(completed_updates=completed,finite=True,ema_updates=completed,
                    optimizer_calls=optimizer_calls,gpu_acceptance=acceptance)
                result['p2b_protocol' if p2b_protocol else 'p2a_protocol'] = paired_protocol
                if p2b_protocol:
                    assert rt.episode_log.record_count == rt.completed == sum(rt.sampler.completed_counts)
                    result['training_episode_records'] = rt.completed
                if args.p3_yanglab_kernel or p3b:
                    result['device_memory']=dict(cuda_allocated_peak_mib=torch.cuda.max_memory_allocated()/2**20,
                                                 cuda_reserved_peak_mib=torch.cuda.max_memory_reserved()/2**20)
                if acceptance == 'rehearsal':
                    counts = np.zeros(4, dtype=np.int64)
                    successes = np.zeros(4, dtype=np.int64)
                    for chunk in rt.episode_log.files:
                        with np.load(rt.episode_log.out/chunk['path'], allow_pickle=False) as data:
                            episodes = data['episodes']
                        regions = rt.pool['region'][episodes['task_id']]
                        counts += np.bincount(regions, minlength=4)
                        successes += np.bincount(regions[episodes['reason']==1], minlength=4)
                    result['region_training'] = {
                        name: dict(episodes=int(n), successes=int(s), success_rate=float(s/n) if n else None)
                        for name, n, s in zip(REGIONS, counts, successes)
                    }
            else:
                for stop in range(100,1501,100):
                    # This RSL release leaves the index at the last completed
                    # iteration; explicitly advance the next chunk's start index.
                    runner.current_learning_iteration=completed
                    runner.learn(num_learning_iterations=100,init_at_random_ep_len=False)
                    assert completed==stop
                    raw={k:v.detach().clone() for k,v in actor.state_dict().items()}
                    torch.save(dict(actor_state_dict=raw,ema_state_dict=ema,completed_updates=completed),args.out/f'actors_{stop:04d}.pt')
                    evaluate_pair(stop,list(range(len(np.load(args.evaluation_pool)['target']))),'evaluations')
                result.update(completed_updates=completed,ema_decay=.99)
            result['first_training_success']=rt.first_training_success
        gpu_phase('shutdown')
        result.update(wall_s=time.monotonic()-start,passed=result.get('passed',True))
        write_json(args.out/'result.json',result)
        print('GOALREACH_RESULT '+json.dumps(result))
    except BaseException:
        fault=traceback.format_exc()
        (args.out/'FAILED.txt').write_text(fault)
        print(fault,flush=True)
        raise
    finally:env.close()


try:
    main()
except BaseException:
    fault=traceback.format_exc()
    (args.out/'FAILED.txt').write_text(fault)
    print(fault,flush=True)
    raise
finally:
    app.close()
