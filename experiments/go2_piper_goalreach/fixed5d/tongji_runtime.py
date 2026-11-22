import fcntl, hashlib, json, os, random
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=(500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,7800,7900,8000)
FULL={1000,2000,3000,4000,5000,6000,7000,7800,7900,8000}
def schedule_for(spec):
    if spec['engineering']:
        return (32,),{32},lambda u,sealed=False: ('ema099',)
    total=spec['updates_per_arm']
    # Half-up rounding gives nearest reachable update at a common interaction fraction.
    inverse={(base*total+4000)//8000:base for base in BASE}
    def modes(update,sealed=False):
        node=inverse[update]
        assert not sealed or node==8000
        return ('raw','ema099') if node%1000==0 and not sealed else ('ema099',)
    return tuple(sorted(inverse)),{u for u,b in inverse.items() if b in FULL},modes
def seed_all(seed):
    import numpy as np, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
def model_digest(values):
    import hashlib
    h=hashlib.sha256()
    for k,v in sorted(values.items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def bind_policy(spec,out,actor,critic):
    import torch
    from protocol import file_sha
    from spatial_train import put
    directory=ROOT/'weights';directory.mkdir(exist_ok=True)
    file=directory/f"seed-{spec['seed']}.pt"
    actual={k:model_digest(v.state_dict()) for k,v in [('actor',actor),('critic',critic)]}
    with file.with_suffix('.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if not file.is_file():
            assert out.parent.name.startswith('benchmark') or spec['selected_arm']=='nsgs_e4_sharp'
            temp=file.with_suffix('.pending')
            torch.save({'actor':{k:v.detach().cpu() for k,v in actor.state_dict().items()},'critic':{k:v.detach().cpu() for k,v in critic.state_dict().items()}},temp)
            temp.replace(file)
        saved=torch.load(file,map_location='cpu',weights_only=True)
        frozen={k:model_digest(saved[k]) for k in ('actor','critic')}
        assert frozen==actual, (frozen,actual)
        spec.update(initial_weights_file=str(file),initial_weights_sha256=file_sha(file),actor_sha256=frozen['actor'])
        put(out/'paired_init.json',dict(seed=spec['seed'],policy=frozen,initial_weights_sha256=spec['initial_weights_sha256']))
def restore_audit(path,runner,rt,ema,updates,calls,out,put):
    import numpy as np, torch
    from protocol import file_sha
    with torch.no_grad():
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['completed_updates']==updates and state['optimizer_calls']==calls
        assert model_digest(runner.alg.actor.state_dict())==model_digest(state['actor'])
        assert model_digest(runner.alg.critic.state_dict())==model_digest(state['critic'])
        assert model_digest(ema)==model_digest(state['ema'])
        before=rt.neural.digest() if rt.neural else None
        previous=rt.sampler.probabilities().copy()
        runner.alg.actor.load_state_dict(state['actor'])
        runner.alg.critic.load_state_dict(state['critic'])
        runner.alg.optimizer.load_state_dict(state['optimizer'])
        if rt.neural: rt.neural.load_state_dict(state['scorer'])
        rt.sampler.load_state_dict(state['sampler'])
        assert np.array_equal(previous,rt.sampler.probabilities())
        if rt.neural: assert before==rt.neural.digest()
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    put(out/'restore.json',dict(checkpoint_sha256=file_sha(path),update=updates,actor=True,critic=True,optimizer=True,ema_verified=True,scorer_and_sampler=True,method='same-process checkpoint reload before independent fixed evaluation and subsequent PPO update'))
def bench_finish(out,rt,env,runner,spec,put):
    import torch
    spec_stop=spec['updates_per_arm']
    budget=json.loads((out/'ppo_budget.json').read_bytes())
    assert budget['ppo_updates_completed']==spec_stop==33
    physical=json.loads((out/'physical_passed.json').read_bytes())
    proof=json.loads((out/'evaluations/0032_switch_audit.json').read_bytes())
    restore=json.loads((out/'restore.json').read_bytes())
    assert proof['rows']=={'ema099':4032} and proof['sampler_isolated'] and proof['actor_restored'] and proof['training_feedback_unchanged'] and restore['update']==32 and physical['controls']==768
    timepoints=[json.loads(line) for line in (out/'timings.jsonl').read_text().splitlines()]
    warm=[t for t in timepoints if t['start']>=20 and t['stop']<=32]
    controls=sum((t['stop']-t['start'])*env.num_envs*24 for t in warm)
    measured=sum(t['training_s'] for t in warm)
    assert measured>0
    put(out/'result.json',dict(passed=True,environment_count=env.num_envs,training_updates=33,controls_per_update=24*env.num_envs,timed_controls=controls,timed_train_seconds=measured,training_controls_per_second=controls/measured,full_ema_seconds=proof['wall_s'],full_ema_rows=4032,after_evaluation_ppo_completed=spec_stop,scorer_updates=rt.neural.estimator.fit_steps if rt.neural else 0,torch_max_reserved_bytes=torch.cuda.max_memory_reserved(),milestones=timepoints))

