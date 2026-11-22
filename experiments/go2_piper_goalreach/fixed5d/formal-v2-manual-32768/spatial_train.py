"""Gated native R0 training with fixed spatial-development and terminal test layouts."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
import random
from time import monotonic
import traceback

from coverage import ROW_KEYS, coverage_point
from protocol import DEVELOPMENT, FULL, VERSION, file_sha, interval_auc, modes, validate


def put(path, data):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    pending.replace(path)


def summarize(rows, pool):
    import numpy as np
    from goalreach.core import REGIONS

    rows.sort(key=lambda row: row['task_id'])
    assert len(rows) == len(pool['task_id'])
    for name in ROW_KEYS:
        assert np.array_equal([row[name] for row in rows],pool[name]), name
    won = np.asarray([row['success'] for row in rows],dtype=np.float64)
    cells = pool['cell_id']
    cell_rates = np.bincount(cells,weights=won,minlength=42)/np.bincount(cells,minlength=42)
    regions = pool['region']
    targets = pool['target_index']
    maps = np.bincount(targets)
    count = np.bincount(targets,weights=won)
    return dict(coverage=float(cell_rates.mean()),cell_rates=cell_rates.tolist(),
                region_rates={name:float(won[regions==i].mean()) for i,name in enumerate(REGIONS)},
                between_condition_disagreement=float(((count>0)&(count<maps)).mean()),
                pair_disagreement=float((2*count*(maps-count)/(maps*(maps-1))).mean()),rows=len(rows))


def evaluate(env, wrapped, runner, ema, panel_path, stop, out, training, spec, *, sealed=False, fixed_ema_only=False):
    import pickle
    import numpy as np
    import torch
    from goalreach import mdp

    def state(rt):
        return sha256(pickle.dumps((rt.sampler.state_dict(),rt.curriculum.state_dict() if rt.curriculum else None),protocol=5)).hexdigest()

    start = monotonic()
    initial = env.cfg.goalreach.copy()
    sampler = state(training)
    scorer = training.neural.digest() if training.neural else None
    before_ends = training.sampler.completed_counts.copy()
    ppo = (out/'ppo_budget.json').read_bytes()
    raw = {key:value.detach().clone() for key,value in runner.alg.actor.state_dict().items()}
    rng = (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    names = ('ema099',) if fixed_ema_only else modes(stop, sealed)
    eval_root = out/('sealed_test' if sealed else 'evaluations')
    eval_root.mkdir(exist_ok=True)
    summaries = {}
    with torch.inference_mode():
        training.discard_at_evaluation_boundary()
        for name in names:
            random.seed(spec['evaluation_seed'])
            np.random.seed(spec['evaluation_seed'])
            torch.manual_seed(spec['evaluation_seed'])
            directory = eval_root/f'{stop:04d}_{name}'
            env.cfg.goalreach.update(pool=str(panel_path),out=str(directory),sampler='evaluation',
                spatial_method='evaluation',binary_log=False,p2b_sgs_log=False,p2a_capture=False,
                k2_episode_metadata=None,training_metrics=False)
            rt = mdp.TaskRuntime(env)
            env.goal_runtime = rt
            assert rt.neural is rt.episode_log is None
            runner.alg.actor.load_state_dict(raw if name=='raw' else ema)
            policy = runner.get_inference_policy(device=spec['learner_device'])
            rows = []
            for offset in range(0,len(rt.pool['task_id']),env.num_envs):
                last = min(offset+env.num_envs,len(rt.pool['task_id']))
                tasks = list(range(offset,last))
                rt.discard_at_evaluation_boundary()
                forced = tasks + [tasks[-1]]*(env.num_envs-len(tasks))
                rt.forced_ids = forced
                rt.eval_repeat = torch.as_tensor(forced,device=env.device)
                wrapped.reset()
                observations = wrapped.get_observations()
                generations = rt.state.generation.tolist()
                seen = set()
                for _ in range(500):
                    actions = policy(observations.to(spec['learner_device']))
                    observations,_,_,_ = wrapped.step(actions.to(env.device))
                    for row in rt.last_rows:
                        slot = row['env_id']
                        if slot < len(tasks) and slot not in seen and row['generation']==generations[slot]:
                            assert row['task_id']==tasks[slot]
                            rows.append(row)
                            seen.add(slot)
                    if len(seen)==len(tasks):
                        break
                assert len(seen)==len(tasks),(stop,name,offset,len(seen))
            assert not any(rt.sampler.completed_counts)
            summaries[name] = summarize(rows,rt.pool)
            put(directory/'rows.json',rows)
            put(directory/'summary.json',summaries[name])
        env.goal_runtime = training
        env.cfg.goalreach = initial
        runner.alg.actor.load_state_dict(raw)
        assert state(training)==sampler and (not training.neural or training.neural.digest()==scorer)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])
        if not sealed:
            wrapped.reset()
        assert np.array_equal(before_ends,training.sampler.completed_counts)
    runner.logger.cur_reward_sum.zero_()
    runner.logger.cur_episode_length.zero_()
    runner.logger.ep_extras = []
    runner.alg.train_mode()
    assert (out/'ppo_budget.json').read_bytes()==ppo
    put(eval_root/f'{stop:04d}_switch_audit.json',dict(update=stop,sampler_isolated=True,scorer_isolated=True,
        ppo_unchanged=True,actor_restored=True,training_feedback_unchanged=True,cumulative_discarded=training.discarded,
        split='test' if sealed else 'development',rows={name:x['rows'] for name,x in summaries.items()},
        row_sources={name:file_sha(eval_root/f'{stop:04d}_{name}'/'rows.json') for name in names},wall_s=monotonic()-start))
    return dict(update=stop,**summaries['ema099'])


def actor_digest(actor):
    h = sha256()
    for key,value in sorted(actor.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def run(args, spec):
    import numpy as np
    import torch
    from goalreach import mdp
    from goalreach.config import agent_cfg, build_cfg, make_env
    from goalreach.core import append_rows
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from spatial_runtime import configure

    global DEVELOPMENT, FULL, modes
    from tongji_runtime import schedule_for
    DEVELOPMENT, FULL, modes = schedule_for(spec)
    arm = args.arm
    out=args.out
    training_file = args.pools/'train.npz'
    simulator = spec.get('simulator_device','cuda:0')
    learner = spec.get('learner_device','cuda:0')
    if simulator=='cpu':
        import warp as wp
        wp.config.kernel_cache_dir=str(out/'warp_cache')
    cfg = build_cfg(training_file,out,spec['num_envs'],seed=spec['seed'],reward='R0',
                    sampler='external' if arm=='nsgs_e4_sharp' else 'uniform',device=simulator,timeout_penalty=-50.)
    configure(cfg,scene_queries=spec.get('scene_queries',True))
    from fixed5d_bench_runtime import configure_bench
    configure_bench(cfg, direct=True, fabric=True)
    cfg.goalreach.update(success_criterion='hold_only',legacy_jsonl=False,spatial_method=arm,p2a_capture=True,
        learner_device=learner,
        binary_log=True,p2b_sgs_log=True,training_metrics=True,training_update=1,sampler_seed=spec['seed'],
        k2_episode_metadata={'stage':'P3B1','arm_id':arm,
                             'spatial_method':arm,'frozen_manifest_sha256':spec['index_manifest_sha256']})
    env = make_env(cfg)
    from packed_origin import check_env
    check_env(env,out)
    try:
        agent = agent_cfg(spec['seed'],.3,.001)
        agent.algorithm.gamma=.99
        assert agent.num_steps_per_env == 24 and env.max_episode_length == 500 and abs(env.step_dt-.02)<1.e-8
        wrapped=RslRlVecEnvWrapper(env,clip_actions=agent.clip_actions)
        for name in ('reset','step','get_observations'):
            setattr(wrapped,name,torch.inference_mode()(getattr(wrapped,name)))
        assert wrapped.get_observations()['policy'].shape == (spec['num_envs'],201)
        rt = mdp.runtime(env)
        if rt.neural and spec.get('scorer_features_sha256'):
            assert rt.neural.feature_metadata['sha256']==spec['scorer_features_sha256']

        class Runner(OnPolicyRunner):
            def save(self,path,infos=None):
                pass

        from tongji_runtime import bind_policy, seed_all
        seed_all(spec['seed'])
        runner = Runner(wrapped,agent.to_dict(),log_dir=str(out/'policy'),device=learner)
        actor = runner.alg.actor
        bind_policy(spec, out, actor, runner.alg.critic)
        if spec.get('initial_weights_sha256'):
            weights=Path(spec['initial_weights_file'])
            assert file_sha(weights)==spec['initial_weights_sha256']
            restored=torch.load(weights,map_location=learner,weights_only=True)
            actor.load_state_dict(restored['actor'])
            runner.alg.critic.load_state_dict(restored['critic'])
        initial = actor_digest(actor)
        put(out/'initial_actor.json',dict(actor_sha256=initial,student_seed=spec['seed'],arm=arm,
                                          native_expected_actor_sha256=spec['actor_sha256'],agent=agent.to_dict()))
        assert initial == spec['actor_sha256']
        assert np.allclose(actor.distribution.log_std_param.detach().exp().cpu().numpy(),.3)
        actual=rt.sampler.probabilities()
        if arm=='manual':
            assert rt.curriculum.stage==0 and not np.any(actual[rt.pool['region']!=0])
            assert np.allclose(actual[rt.pool['region']==0],1/np.sum(rt.pool['region']==0),rtol=1e-14)
        else:
            assert np.array_equal(actual,np.full(len(actual),1/len(actual)))
        ema = {key:value.detach().clone() for key,value in actor.state_dict().items()}
        if args.audit_entry:
            assert rt.neural is not None and rt.neural.estimator.fit_steps == 0
            fixed = runner.get_inference_policy(device=learner)
            with torch.inference_mode():
                obs = wrapped.get_observations()
                for _ in range(6):
                    actions = fixed(obs.to(learner))
                    assert torch.isfinite(actions).all()
                    obs,_,_,_ = wrapped.step(actions.to(env.device))
                assert len([key for key in rt.planes.keys if key is not None])==spec['num_envs']

            class RolloutReady(Exception):
                pass

            def stop_before_optimization():
                raise RolloutReady

            runner.alg.update = stop_before_optimization
            try:
                runner.learn(num_learning_iterations=1,init_at_random_ep_len=False)
            except RolloutReady:
                pass
            else:
                raise RuntimeError('Native runner did not reach its PPO optimizer boundary')
            put(out/'passed.json',dict(passed=True,entry_only=True,initial_actor_sha256=initial,
                slots=env.num_envs,scorer_features=rt.neural.feature_metadata,ppo_updates=0,
                simulator_device=str(env.device),learner_device=str(next(actor.parameters()).device),
                scorer_device=str(rt.neural.estimator.device),
                scorer_optimizer_steps=0,native_rollout_steps=24,geometry_batches=rt.planes.counters))
            return
        from formal_gpu4096_runtime import attest
        attest(args,out,rt,env,wrapped,runner,spec,put)
        updates,calls,scorer_passed = 0,0,0
        original_update,original_step = runner.alg.update,runner.alg.optimizer.step

        def budget(started,complete):
            put(out/'ppo_budget.json',dict(ppo_updates_started=started,ppo_updates_completed=complete,
                optimizer_calls=calls,student_updates_executed=updates,scorer_updates_passed=scorer_passed))

        def checkpoint_state(path, *, diagnostic=False):
            pending = path.with_suffix('.pending')
            state = dict(actor=actor.state_dict(),critic=runner.alg.critic.state_dict(),
                optimizer=runner.alg.optimizer.state_dict(),ema=ema,sampler=rt.sampler.state_dict(),curriculum=rt.curriculum.state_dict() if rt.curriculum else None,
                scorer=rt.neural.state_dict() if rt.neural else None,completed_updates=updates,
                optimizer_calls=calls,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                numpy_rng=np.random.get_state(),python_rng=random.getstate())
            if diagnostic:
                state.update(snapshot_schema='spatial-probability-checkpoint-v1',
                             purpose='diagnostic',resumption_validated=False,scorer_updates_passed=scorer_passed)
            torch.save(state,pending)
            pending.replace(path)
            if diagnostic:
                put(path.with_suffix('.json'),dict(schema=state['snapshot_schema'],
                    file=path.name,sha256=file_sha(path),student_updates_executed=updates,
                    optimizer_calls=calls,scorer_updates_passed=scorer_passed,purpose='diagnostic',
                    resumption_validated=False))
    
        def optimizer_step(*a,**kw):
            nonlocal calls
            calls += 1
            return original_step(*a,**kw)

        def update():
            nonlocal updates, scorer_passed
            budget(updates+1,updates)
            opt_start=monotonic()
            losses = original_update()
            opt_end=monotonic()
            updates += 1
            budget(updates,scorer_passed)
            assert calls == updates*20 and all(np.isfinite(float(value)) for value in losses.values())
            assert all(torch.isfinite(value).all() for value in (*actor.parameters(),*runner.alg.critic.parameters()))
            with torch.no_grad():
                for key,value in actor.state_dict().items():
                    if value.is_floating_point():
                        ema[key].mul_(.99).add_(value.detach(),alpha=.01)
                    else:
                        ema[key].copy_(value)
            try:
                scorer_start=monotonic()
                rt.end_update(updates)
                scorer_end=monotonic()
                if spec['engineering']: append_rows(out/'optimizer-timing.jsonl',[dict(update=updates,ppo_s=opt_end-opt_start,scorer_s=scorer_end-scorer_start)])
            except (AssertionError, ValueError, FloatingPointError) as error:
                if getattr(rt, 'scorer_fault_update', None)==updates:
                    try:
                        checkpoint_state(out/f'checkpoint_probability_fault_{updates:04d}.pt',diagnostic=True)
                    except Exception as snapshot_error:
                        error.add_note(f'Diagnostic checkpoint could not be fully saved: {snapshot_error!r}')
                    try:
                        rt.episode_log.flush(updates,final=True)
                    except Exception as log_error:
                        error.add_note(f'Natural ending ledger could not be flushed: {log_error!r}')
                raise
            env.cfg.goalreach['training_update']=updates+1
            scorer_passed=updates
            budget(updates,updates)
            if updates<=2 or updates==101 or updates%100==0 or updates-1 in DEVELOPMENT:
                append_rows(out/'milestones.jsonl',[dict(update=updates,optimizer_calls=calls,
                    losses={key:float(value) for key,value in losses.items()},pid=__import__('os').getpid())])
            return losses

        runner.alg.optimizer.step = optimizer_step
        runner.alg.update = update
        budget(0,0)
        evaluation_progress=out/'evaluation_progress.json'
        put(evaluation_progress,dict(development_evaluated_through=0,terminal_evaluated=False))
        light_curve = []
        full_curve = []
        start = monotonic()
        if args.learning_probe:
            from learning_probe import run_probe
            run_probe(args,out,rt,env,wrapped,runner,ema,spec)
            return
        if args.grid_development:
            from grid_development import run_grid_development
            run_grid_development(args,out,rt,env,wrapped,runner,ema,spec)
            return
        total = spec['updates_per_arm']
        for stop in sorted(set(range(100,total+1,100)) | set(DEVELOPMENT) | ({20} if spec['engineering'] else set()) | {total}):
            prior=updates
            batch_start=monotonic()
            runner.current_learning_iteration = updates
            runner.learn(num_learning_iterations=stop-updates,init_at_random_ep_len=False)
            assert updates==stop
            append_rows(out/'timings.jsonl',[dict(start=prior,stop=stop,training_s=monotonic()-batch_start,envs=env.num_envs)])
            rt.episode_log.flush(stop,final=stop==total)
            if stop not in DEVELOPMENT:
                append_rows(out/'skipped_evaluation_boundaries.jsonl',[dict(update=stop,
                    active_episodes=int((rt.state.task>=0).sum()),control_steps=env.common_step_counter,
                    cumulative_discarded=rt.discarded,optimizer_calls=calls)])
                continue
            checkpoint = out/(f'checkpoint_{stop:04d}.pt' if stop%500==0 else 'checkpoint_boundary.pt')
            checkpoint_state(checkpoint)
            if spec['engineering'] and stop==32:
                from tongji_runtime import restore_audit
                restore_audit(checkpoint,runner,rt,ema,updates,calls,out,put)
            full=stop in FULL
            panel=args.pools/'development.npz' if full else args.compact_panel
            metric=evaluate(env,wrapped,runner,ema,panel,stop,out,rt,spec)
            if full:
                full_curve.append(metric)
                put(out/'evaluations'/'ema_full_nodes.json',full_curve)
            light_curve.append(coverage_point(out,stop,args.original_mapping,compact=not full))
            put(out/'evaluations'/'ema_curve_504_sparse.json',light_curve)
            put(evaluation_progress,dict(development_evaluated_through=stop,terminal_evaluated=False))
        if spec['engineering']:
            from tongji_runtime import bench_finish
            bench_finish(out,rt,env,runner,spec,put)
            return
        test = evaluate(env,wrapped,runner,ema,args.pools/'test.npz',total,out,rt,spec,sealed=True)
        put(evaluation_progress,dict(development_evaluated_through=total,terminal_evaluated=True))
        last = full_curve[-3:]
        assert [row['update'] for row in last]==sorted(FULL)[-3:]
        assert int(rt.sampler.completed_counts.sum()) == rt.episode_log.record_count
        assert int(rt.sampler.sample_counts.sum()) == int(rt.sampler.completed_counts.sum())+rt.discarded
        put(out/'result.json',dict(passed=True,arm=arm,seed=spec['seed'],initial_actor_sha256=initial,updates=updates,
            optimizer_calls=calls,transitions=updates*24*env.num_envs,wall_s=monotonic()-start,
            evaluation_version=VERSION,schedule_sha256=args.schedule_sha,
            development_auc_504_sparse=sum((b['update']-a['update'])*(a['coverage']+b['coverage'])/2 for a,b in zip(light_curve,light_curve[1:]))/(total-DEVELOPMENT[0]),development_auc_interval=[DEVELOPMENT[0],total],
            development_auc_normalization_updates=total-DEVELOPMENT[0],
            endpoint_ema_coverage=float(np.mean([row['coverage'] for row in last])),
            endpoint_ema_walk_rate=float(np.mean([row['region_rates']['walk'] for row in last])),sealed_test=test))
    finally:
        env.close()


