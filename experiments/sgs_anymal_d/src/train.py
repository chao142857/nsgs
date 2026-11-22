"""Single-seed fixed-scale paper replication. Only the tabular SGS sampler is used."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT.parents[1]
TOTAL = 5000
EVAL_NODES = {250,500,1000,2000,3000,4000,5000}
SLOTS = 32768


def train(world,args):
    import torch
    from diagnostics import qualify, evaluate
    from ppo_smoke import make_algorithm, obs, fingerprint, save, append, STEPS
    from sampler import Sampler

    out=args.out
    world.num_envs,world.num_actions=SLOTS,12
    boot=out if not args.resume_at else out/'attempts'/f'resume-u{args.resume_at:04d}-{os.getpid()}'
    boot.mkdir(parents=True,exist_ok=True)
    save(boot/'scene.json',world.probe_geometry())
    qualify(world,boot)
    assert STEPS==24
    random.seed(args.seed);torch.manual_seed(args.seed);np.random.seed(args.seed)
    sampler=Sampler(args.seed,world.device)
    initial=sampler.sample(SLOTS)
    world.place(world.env_ids,initial,True)
    observations=obs(world)
    alg=make_algorithm(world,observations)
    initial_state=dict(seed=args.seed,card=args.card,actual_envs=world.robot.num_instances,
         actor_sha256=fingerprint(alg.actor.state_dict()),critic_sha256=fingerprint(alg.critic.state_dict()),
         initial_reset_sha256=fingerprint(initial),from_scratch=True)
    if args.resume_at:
        assert initial_state==json.loads((out/'initial.json').read_text()), 'Wrong initial policy or seed'
    else:
        save(out/'initial.json',initial_state)
    natural=np.zeros(5,np.int64)
    windows=[]
    if args.resume_at:
        start=args.resume_at
        path=out/f'checkpoint-{start:04d}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False)
        log=[json.loads(row) for row in (out/'updates.jsonl').read_text().splitlines()]
        assert [v['update'] for v in log]==list(range(1,start+1))
        assert ck['seed']==args.seed and ck['update']==start
        assert ck['controls']==log[-1]['controls']==start*SLOTS*STEPS
        natural=ck['natural_outcomes'].copy()
        assert np.array_equal(natural,log[-1]['outcomes'])
        boundary=json.loads((out/'evaluation_boundaries.jsonl').read_text().splitlines()[-1])
        assert boundary==dict(update=start,censored_open=SLOTS,training_labels=int(natural.sum()))
        summary=json.loads((out/'evaluations'/f'u{start:04d}'/'summary.json').read_text())
        assert summary['seed']==args.seed and summary['update']==start
        alg.load(ck['ppo'],load_cfg=None,strict=True)
        alg.learning_rate=ck['learning_rate']
        for key,state in alg.save().items():
            assert fingerprint(state)==fingerprint(ck['ppo'][key]), key
        assert fingerprint(alg.actor.state_dict())==summary['actor_sha256']
        sampler.table.load(ck['table'])
        for k,v in sampler.table.state().items():assert np.array_equal(v,ck['table'][k]),k
        sampler.refresh()
        assert int(sampler.table.seen.sum())==int(natural.sum())
        sampler.rng.set_state(ck['sampler_rng'].cpu())
        planned=sampler.sample(SLOTS)
        sampler.rng.set_state(ck['sampler_rng'].cpu())
        reset_ids=sampler.sample(SLOTS)
        assert torch.equal(planned,reset_ids)
        torch.set_rng_state(ck['torch_rng'].cpu())
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu(),world.device)
        assert torch.equal(torch.get_rng_state(),ck['torch_rng'].cpu())
        assert torch.equal(torch.cuda.get_rng_state(world.device),ck['cuda_rng'].cpu())
        world.place(world.env_ids,reset_ids,True)
        observations=obs(world)
        save(boot/'restored.json',dict(seed=args.seed,card=args.card,update=start,
             sha256=hashlib.sha256(path.read_bytes()).hexdigest(),initial_verified=True,
             actor_verified=True,critic_verified=True,optimizer_verified=True,sgs_table_verified=True,
             learning_rate=alg.learning_rate,reset_ids_sha256=fingerprint(reset_ids),labels=int(natural.sum()),
             controls=ck['controls'],cuda_rng_verified=True,open_slots_reset_at_recorded_evaluation_boundary=SLOTS))
        del ck
    for update in range(args.resume_at+1,TOTAL+1):
        clock=time.monotonic()
        first_weight=fingerprint(alg.actor.state_dict()) if update==1 else None
        alg.train_mode()
        with torch.no_grad():
            for step in range(STEPS):
                action=alg.act(observations)
                labels,signals=world.step(action,training_iteration=update-1)
                next_obs=obs(world); done=labels!=0
                slots=done.nonzero().flatten()
                reward=signals['reward']
                if slots.numel():
                    truncated=labels==4
                    reward=reward+alg.gamma*alg.critic(next_obs).squeeze(-1)*truncated
                    previous=world.task_ids[slots].cpu().numpy()
                    outcomes=labels[slots].cpu().numpy()
                    natural+=np.bincount(outcomes,minlength=5)
                    sampler.record(previous,outcomes==1)
                    next_ids=sampler.sample(len(previous))
                    windows.append(np.column_stack((np.full(len(previous),update,np.int32),
                        np.full(len(previous),step,np.int32),previous,outcomes,next_ids.cpu().numpy())).astype(np.int32))
                    world.place(slots,next_ids)
                    next_obs=obs(world)
                alg.process_env_step(next_obs,reward,done,{})
                observations=next_obs
            alg.compute_returns(observations)
        calls=[0]
        def increment(*_): calls[0]+=1
        h=alg.optimizer.register_step_post_hook(increment)
        try: losses=alg.update()
        finally:h.remove()
        torch.cuda.synchronize()
        assert calls[0]==20 and all(np.isfinite(float(v)) for v in losses.values())
        if update==1: assert fingerprint(alg.actor.state_dict())!=first_weight
        append(out/'updates.jsonl',dict(update=update,seed=args.seed,envs=SLOTS,controls=update*SLOTS*STEPS,
            ppo_optimizer_steps=calls[0],sgs_labels=int(sampler.table.seen.sum()),outcomes=natural.tolist(),
            wall_s=time.monotonic()-clock,losses={k:float(v) for k,v in losses.items()}))
        assert sampler.table.seen.sum()==natural.sum()
        if update%100==0 or update in EVAL_NODES:
            if windows:
                dest=out/f'episodes-through-{update:04d}.npz'
                with dest.with_suffix('.part').open('wb') as stream:
                    np.savez_compressed(stream,episodes=np.concatenate(windows))
                dest.with_suffix('.part').replace(dest)
                windows.clear()
            savefile=out/f'checkpoint-{update:04d}.pt'
            tmp=savefile.with_suffix('.pending')
            torch.save(dict(ppo=alg.save(),seed=args.seed,update=update,table=sampler.table.state(),
                sampler_rng=sampler.rng.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(world.device),natural_outcomes=natural,
                learning_rate=alg.learning_rate,controls=update*SLOTS*STEPS),tmp)
            tmp.replace(savefile)
        if update in EVAL_NODES:
            before=fingerprint(alg.actor.state_dict())
            history=int(sampler.table.seen.sum())
            evaluate(world,alg,out,update,args.seed)
            assert before==fingerprint(alg.actor.state_dict()) and int(sampler.table.seen.sum())==history
            # Censor unfinished training episodes at evaluation boundaries.
            append(out/'evaluation_boundaries.jsonl',dict(update=update,censored_open=SLOTS,training_labels=history))
            if update<TOTAL:
                world.place(world.env_ids,sampler.sample(SLOTS))
                observations=obs(world)
                append(out/'post_eval_reset.jsonl',dict(update=update,mutable_reset_completed=True))
    save(out/'result.json',dict(completed=TOTAL,envs=SLOTS,seed=args.seed,
        interactions=TOTAL*SLOTS*STEPS,labels=int(natural.sum()),final_heldout=json.loads(
            (out/'evaluations/u5000/summary.json').read_text())))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('card',type=int,choices=range(4));parser.add_argument('seed',type=int)
    parser.add_argument('out',type=Path)
    parser.add_argument('--resume-at',type=int,default=0)
    args=parser.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    manifest=json.loads((ROOT/'protocol.json').read_text())
    assert manifest['seeds'][args.card]==args.seed and manifest['envs']==SLOTS and manifest['updates']==TOTAL
    assert args.out.resolve().is_relative_to(ROOT/'runs')
    assert args.resume_at in ({0}|(EVAL_NODES-{TOTAL}))
    assert (args.out/'initial.json').is_file()==bool(args.resume_at)
    kit=['--portable-root',str(args.out/'portable'),'--/app/extensions/registryEnabled=false',
        '--/renderer/multiGpu/enabled=false','--/renderer/multiGpu/autoEnable=false',
        f'--/renderer/activeGpu={args.card}','--/physics/cudaDevice=0',
        '--/plugins/carb.tasking.plugin/threadCount=2','--/plugins/omni.tbb.globalcontrol/maxThreadCount=2',
        f'--/log/file={args.out}/kit.log']
    kit.extend(f'--/persistent/isaac/asset_root/{k}={ROOT}/assets' for k in ('local','cloud','default','nvidia'))
    sys.argv=sys.argv[:1]
    from isaaclab.app import AppLauncher
    class Bound(AppLauncher):
        def _resolve_device_settings(self,config):
            super()._resolve_device_settings(config)
            config.update(active_gpu=args.card,physics_gpu=0,multi_gpu=False,limit_cpu_threads=2)
    app=Bound(headless=True,device='cuda:0',enable_cameras=False,livestream=0,
        experience=str(PROJECT/'src/static/lab/apps/isaaclab.python.headless.kit'),kit_args=' '.join(kit)).app
    try:
        from native import BatchedScene
        world=BatchedScene(SLOTS)
        train(world,args)
    except BaseException:
        (args.out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__':
    main()
