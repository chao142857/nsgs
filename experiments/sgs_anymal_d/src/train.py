"""Bounded, single-card SGS pilot. Evaluations replay immutable checkpoints separately."""
import argparse
from hashlib import sha256
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import random
import sys
from time import monotonic
import traceback

import numpy as np
from catalog import ROOT
from pilot_config import (CLASS_NAMES, atomic_json, check_physics, file_sha, identity, locked_budget,
                          reserve, read)
from ppo_smoke import append, fingerprint, make_algorithm, obs, STEPS


def record_groups(terrain_ids, classes, ids, labels=None):
    groups=terrain_ids[ids]*3 + classes[ids]
    if labels is None:
        return np.bincount(groups,minlength=39).reshape(13,3)
    return np.bincount(groups*5+labels,minlength=195).reshape(13,3,5)


def experiment_inputs(out, world, item, physics_sha):
    modules={}
    for name in item['source_sha256']:
        if name=='train.py':
            path=Path(__file__).resolve()
        else:
            path=Path(import_module(name[:-3]).__file__).resolve()
        if file_sha(path)!=item['source_sha256'][name]:
            raise ValueError(f'Runtime imported a different {name}: {path}')
        modules[name]=str(path)
    state=world.robot.data
    components=dict(physics_dt_s=world.sim.get_physics_dt(),control_decimation=4,
                    joints=world.robot.joint_names,feet=world.foot_names,
                    actuators={name:type(model).__module__+'.'+type(model).__name__ for name,model in world.robot.actuators.items()},
                    policy_signal='body velocity, projected gravity, joint deltas, joint velocities, last action, body-frame target, yaw-aligned height scan',
                    rays=world.scan.data.ray_hits_w.shape[1],obs_size=world.observations().shape[1],
                    action_size=state.joint_pos.shape[1],default_joint_positions=state.default_joint_pos[0].tolist(),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    packages={}
    for name in ('rsl-rl-lib','torch','isaaclab','isaacsim','numpy'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    atomic_json(out/'loaded.json',dict(identity=item,modules=modules,packages=packages,physics_receipt_sha256=physics_sha,
                                      components=components,training_truncation_for_evaluation=0))


def check_history(sampler, outcomes):
    if int(sampler.table.seen.sum())!=int(outcomes.sum()) or int(sampler.table.success.sum())!=int(outcomes[1]):
        raise RuntimeError('SGS history and natural training outcomes diverged')


def checkpoint(file, state):
    import torch
    pending=file.with_suffix('.pending')
    with pending.open('wb') as stream:
        torch.save(state,stream)
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(file)


def next_attempt(out, resume):
    directory=out/'attempts'
    directory.mkdir(exist_ok=True)
    existing=sorted(directory.glob('attempt-*'))
    if not resume and any(out.glob('checkpoint-*.pt')):
        raise ValueError('Output already has a checkpoint; explicitly select the latest one with --resume-at')
    attempt=directory/f'attempt-{len(existing)+1:03d}'
    attempt.mkdir()
    return attempt


def train(world,args,item,attempt,budget_path,budget,clock):
    import torch
    import navigation as nav
    from sampler import Sampler
    out=args.out
    n=args.num_envs
    world.num_envs,world.num_actions=n,12
    world.select_bank('train')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sampler=Sampler(args.seed,world.device)
    initial=sampler.sample(n)
    world.place(world.env_ids,initial,verify=True)
    readings=obs(world)
    alg=make_algorithm(world,readings,args.updates)
    initial_state=dict(seed=args.seed,actor_sha256=fingerprint(alg.actor.state_dict()),
                       critic_sha256=fingerprint(alg.critic.state_dict()),optimizer_sha256=fingerprint(alg.optimizer.state_dict()),
                       optimizer_states=len(alg.optimizer.state),sgs_seen=int(sampler.table.seen.sum()),
                       uniform_p0_mass=bool(np.all(sampler.table.mass==sampler.table.mass[0])),
                       reset_sha256=fingerprint(initial),num_envs=n,from_scratch=True)
    if initial_state['sgs_seen'] or initial_state['optimizer_states'] or not initial_state['uniform_p0_mass']:
        raise RuntimeError('Fresh initialization did not start with empty optimizer and SGS histories')
    atomic_json(attempt/'initial.json',initial_state)
    classes=np.load(ROOT/'train_classes.npy',allow_pickle=False).astype(np.int32)
    terrain_ids=np.arange(len(classes),dtype=np.int32)//8000
    routes=np.load(ROOT/'train_route_types.npy',allow_pickle=False).astype(np.int32)
    route_names=read(ROOT/'catalog.json')['route_types']
    route_draws=np.bincount(routes[initial.cpu().numpy()],minlength=len(route_names))
    route_labels=np.zeros((len(route_names),5),dtype=np.int64)
    outcomes=np.zeros(5,np.int64)
    recent_draws=record_groups(terrain_ids,classes,initial.cpu().numpy())
    recent_labels=np.zeros((13,3,5),np.int64)
    cumulative_controls=0
    previous_seconds=0.
    start=args.resume_at
    if args.resume:
        latest=max((int(p.stem.split('-')[-1]) for p in out.glob('checkpoint-*.pt')),default=0)
        if start!=latest or start>=args.updates:
            raise ValueError('Only the latest incomplete same-version checkpoint is eligible to resume')
        file=out/f'checkpoint-{start:04d}.pt'
        ck=torch.load(file,map_location='cpu',weights_only=False)
        if ck['identity']!=item or ck['initial']!=initial_state or ck['update']!=start or ck['budget_reserved']>budget['reserved']:
            raise ValueError('Wrong initialization, budget lineage, task bank or configuration')
        alg.load(ck['ppo'],load_cfg=None,strict=True)
        for key,value in alg.save().items():
            if fingerprint(value)!=fingerprint(ck['ppo'][key]):
                raise ValueError(f'Checkpoint restoration mismatch for {key}')
        alg.learning_rate=ck['learning_rate']
        sampler.table.load(ck['table']);sampler.refresh()
        for key,value in sampler.table.state().items():
            np.testing.assert_array_equal(value,ck['table'][key])
        sampler.rng.set_state(ck['sampler_rng'].cpu())
        torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),world.device)
        np.random.set_state(ck['numpy_rng'])
        random.setstate(ck['python_rng'])
        outcomes=ck['natural_outcomes'].copy()
        check_history(sampler,outcomes)
        # At u0 restore the initial reset itself, whose IDs are already reflected in the saved RNG.
        reset_ids=sampler.sample(n) if start else initial
        world.place(world.env_ids,reset_ids,True)
        readings=obs(world)
        cumulative_controls=ck['controls']
        previous_seconds=ck['run_wall_s']
        recent_draws=record_groups(terrain_ids,classes,reset_ids.cpu().numpy())
        route_draws=np.bincount(routes[reset_ids.cpu().numpy()],minlength=len(route_names))
        atomic_json(attempt/'resume.json',dict(checkpoint_sha256=file_sha(file),from_update=start,
                    censored_open_on_recovery=n if start else 0,sgs_feedback_from_censored=0,eval_censored=0,
                    snapshot_order='policy/optimizer/SGS/RNG restored, active physics episodes explicitly discarded',
                    original_budget_reserved=ck['budget_reserved'],charged_so_far=budget['reserved']))
        del ck
    else:
        atomic_json(out/'initial.json',initial_state)
        checkpoint(out/'checkpoint-0000.pt',dict(ppo=alg.save(),seed=args.seed,update=0,controls=0,
            identity=item,initial=initial_state,table=sampler.table.state(),sampler_rng=sampler.rng.get_state(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(world.device),
            numpy_rng=np.random.get_state(),python_rng=random.getstate(),natural_outcomes=outcomes,
            learning_rate=alg.learning_rate,run_wall_s=monotonic()-clock,budget_reserved=budget['reserved']))
    np.save(attempt/'initial_assignment.npy', (reset_ids if args.resume else initial).cpu().numpy(),allow_pickle=False)
    windows=[]
    last_update=start
    window_from=start+1
    for update in range(start+1,args.updates+1):
        if budget['reserved']>=budget['cap']:break
        tick=monotonic()
        before=fingerprint(alg.actor.state_dict()) if update==1 else None
        reward_sums=torch.zeros(7,device=world.device,dtype=torch.float64)
        objective_sum=torch.zeros((),device=world.device,dtype=torch.float64)
        adjustment_sum=torch.zeros_like(objective_sum)
        alg.train_mode()
        with torch.no_grad():
            for step in range(STEPS):
                action=alg.act(readings)
                labels,sig=world.step(action,training_iteration=update-1)
                next_obs=obs(world)
                done=labels.ne(0)
                slots=done.nonzero().flatten()
                raw=sig['reward']
                objective_sum+=raw.sum(dtype=torch.float64)
                reward_sums+=sig['reward_terms'].sum(dim=0,dtype=torch.float64)
                reward=raw
                if slots.numel():
                    bootstrap=alg.gamma*alg.critic(next_obs).squeeze(-1)*(labels==4)
                    adjustment_sum+=bootstrap.sum(dtype=torch.float64)
                    reward=raw+bootstrap
                    completed_ids=world.task_ids[slots].cpu().numpy()
                    natural=labels[slots].cpu().numpy()
                    outcomes+=np.bincount(natural,minlength=5)
                    sampler.record(completed_ids,natural==1)
                    next_ids=sampler.sample(len(completed_ids))
                    draws=next_ids.cpu().numpy()
                    recent_draws+=record_groups(terrain_ids,classes,draws)
                    recent_labels+=record_groups(terrain_ids,classes,completed_ids,natural)
                    route_draws+=np.bincount(routes[draws],minlength=len(route_names))
                    route_labels+=np.bincount(routes[completed_ids]*5+natural,
                                              minlength=len(route_names)*5).reshape(-1,5)
                    windows.append(np.column_stack((np.full(len(draws),update),np.full(len(draws),step),
                                  slots.cpu().numpy(),completed_ids,natural,draws)).astype(np.int32))
                    world.place(slots,next_ids)
                    next_obs=obs(world)
                alg.process_env_step(next_obs,reward,done,{})
                readings=next_obs
            alg.compute_returns(readings)
        calls=[0]
        def count_step(*_):
            calls[0]+=1
        if not reserve(budget_path,budget):raise RuntimeError('Concurrent budget modification')
        hook=alg.optimizer.register_step_post_hook(count_step)
        try:losses=alg.update()
        finally:hook.remove()
        torch.cuda.synchronize()
        if calls[0]!=20 or not all(np.isfinite(float(v)) for v in losses.values()):
            raise RuntimeError(f'Invalid PPO update {update}, counted {calls[0]} optimizer steps')
        if update==1 and before==fingerprint(alg.actor.state_dict()):
            raise RuntimeError('First PPO update left the actor unchanged')
        budget['completed']+=1
        atomic_json(budget_path,budget)
        cumulative_controls+=n*STEPS
        check_history(sampler,outcomes)
        sums=reward_sums.cpu().numpy()
        task_sum=float(objective_sum)
        if abs(float(sums.sum())-task_sum)>3e-6*max(1.,float(abs(sums).sum())):
            raise RuntimeError('Logged reward terms do not sum to the training reward')
        last_update=update
        wall_s=previous_seconds+monotonic()-clock
        row=dict(update=update,seed=args.seed,num_envs=n,controls=cumulative_controls,ppo_optimizer_steps=calls[0],
             budget_charged=budget['reserved'],sgs_labels=int(sampler.table.seen.sum()),outcomes=outcomes.tolist(),
             rollout_and_ppo_seconds=monotonic()-tick,single_gpu_elapsed_hours=wall_s/3600.,
             reward_term_sums=sums.tolist(),reward_weights=nav.reward_weights(update-1),
             sum_environment_reward=task_sum,sum_timeout_value_bootstrap=float(adjustment_sum),
             losses={k:float(v) for k,v in losses.items()})
        append(attempt/'updates.jsonl',row)
        if update==1:
            atomic_json(out/'first-update.json',dict(**row,bank=item['geometry_hashes'][0],manifest=item['manifest'],
                        actor_initial=before,actor_after=fingerprint(alg.actor.state_dict()),source_git=item['git_commit']))
        flush=update%25==0 or update in args.eval_at or budget['reserved']==budget['cap']
        if flush:
            if windows:
                episode_file=attempt/f'episodes-{window_from:04d}-{update:04d}.npz'
                with episode_file.with_suffix('.part').open('wb') as stream:
                    np.savez_compressed(stream,episodes=np.concatenate(windows))
                episode_file.with_suffix('.part').replace(episode_file)
                windows.clear()
            window=dict(begin=window_from,end=update,class_names=CLASS_NAMES,axes=['terrain','class'],
                draws=recent_draws.tolist(),ended_by_label=recent_labels.tolist(),
                draw_fraction_of_all=(recent_draws/max(1,int(recent_draws.sum()))).tolist(),
                success_over_ended=[[float(x[1]/x.sum()) if x.sum() else None for x in group] for group in recent_labels])
            window['route_types']={name:dict(drawn=int(count),draw_fraction=float(count/max(1,route_draws.sum())),
                ended=int(labels.sum()),success=int(labels[1]),
                success_over_ended=float(labels[1]/labels.sum()) if labels.sum() else None)
                for name,count,labels in zip(route_names,route_draws,route_labels)}
            if int(route_draws.sum())!=int(recent_draws.sum()) or int(route_labels.sum())!=int(recent_labels.sum()):
                raise ValueError('Route subdivisions do not reconcile')
            append(attempt/'route-windows.jsonl',window)
            route_draws.fill(0);route_labels.fill(0)
            recent_draws.fill(0);recent_labels.fill(0);window_from=update+1
            file=out/f'checkpoint-{update:04d}.pt'
            if file.exists():raise FileExistsError(file)
            checkpoint(file,dict(ppo=alg.save(),seed=args.seed,update=update,controls=cumulative_controls,
                identity=item,initial=initial_state,table=sampler.table.state(),sampler_rng=sampler.rng.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(world.device),
                numpy_rng=np.random.get_state(),python_rng=random.getstate(),natural_outcomes=outcomes,
                learning_rate=alg.learning_rate,run_wall_s=wall_s,budget_reserved=budget['reserved']))
            if update in args.eval_at:
                atomic_json(out/f'evaluate-u{update:04d}.json',dict(update=update,seed=args.seed,file=file.name,
                    checkpoint_sha256=file_sha(file),manifest=item['manifest'],training_feedback=0,
                    training_censored_for_evaluation=0,mode='independent checkpoint replay'))
    atomic_json(out/'result.json',dict(completed_updates=last_update,requested_updates=args.updates,
        seed=args.seed,num_envs=n,controls=cumulative_controls,charged=budget['reserved'],cap=budget['cap'],
        status='completed' if last_update==args.updates else 'cumulative_budget_exhausted',natural_outcomes=outcomes.tolist(),
        training_labels=int(outcomes.sum()),training_truncated_for_evaluations=0,
        open_censored_at_end=n,eval_at=args.eval_at,single_gpu_elapsed_hours=(previous_seconds+monotonic()-clock)/3600))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--num-envs',type=int,default=4096)
    p.add_argument('--updates',type=int,default=500)
    p.add_argument('--eval-at',type=int,nargs='+',default=[250,500])
    p.add_argument('--physics-receipt',type=Path,required=True)
    p.add_argument('--resume-at',type=int,help='resume the latest same-version checkpoint, including zero')
    p.add_argument('--budget-ledger',type=Path,help='shared across attempts; default next to output directory')
    p.add_argument('--device-index',type=int,default=0,help='physical Kit graphics index; CUDA_VISIBLE_DEVICES binds physics')
    return p.parse_args()


def main():
    t=monotonic()
    args=parse_args()
    args.out=args.out.resolve()
    args.resume=args.resume_at is not None
    args.resume_at=args.resume_at or 0
    if args.resume_at<0 or args.resume_at>=args.updates or not 0<=args.seed<2**32-1:
        raise ValueError('Invalid resume update or seed')
    item=identity(ROOT,args)
    physics_sha=check_physics(args.physics_receipt,item)
    ledger=args.budget_ledger or args.out.parent/'pilot-budget.json'
    with locked_budget(ledger,item,args.out,args.resume) as (ledger,budget):
        args.out.mkdir(parents=True,exist_ok=True)
        identity_file=args.out/'identity.json'
        if identity_file.exists() and read(identity_file)!=item:
            raise ValueError('Output directory belongs to another configuration')
        atomic_json(identity_file,item)
        attempt=next_attempt(args.out,args.resume)
        from isaac_entry import open_app
        app=open_app(args.device_index,attempt)
        try:
            from native import BatchedScene
            world=BatchedScene(args.num_envs)
            experiment_inputs(attempt,world,item,physics_sha)
            train(world,args,item,attempt,ledger,budget,t)
        except BaseException:
            (attempt/'error.txt').write_text(traceback.format_exc())
            raise
        finally:
            app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':
    main()
