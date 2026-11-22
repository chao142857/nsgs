"""Run real RSL-RL PPO on the validated ANYmal-D task dictionary."""

from argparse import ArgumentParser
from collections import Counter
from datetime import datetime
from hashlib import sha256
from importlib.metadata import version
import json
import math
from pathlib import Path
import random
import time
import traceback
from zoneinfo import ZoneInfo

import navigation as nav
from scene import GoalScene, LABELS
from terrain import GEOMETRY_VERSION, LEVELS, TASK_COUNT, TERRAIN_NAMES, TERRAIN_SEEDS, decode, encode


EDITION = "sgs-anymald-ppo-smoke-r3"
SEED = 47019
EVAL_SEED = 47207
STEPS = 24
RUN_UPDATES = 6
TARGET, KAPPA, EPS, TEMPERATURE, HISTORY = .66, 5., 1e-8, 2., 100


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def save(path, value):
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def append(path, value):
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def fingerprint(tree):
    import torch

    h = sha256()

    def visit(value):
        if torch.is_tensor(value):
            h.update(repr((str(value.dtype), tuple(value.shape))).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=repr):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            h.update(str(len(value)).encode())
            for entry in value:
                visit(entry)
        else:
            h.update(repr(value).encode())
        h.update(b"\x00")

    visit(tree)
    return h.hexdigest()


class Sampling:
    def __init__(self, torch, method, device):
        from goalreach.vendor_sampler import TaskSampler

        self.torch, self.device, self.method = torch, device, method
        self.vendor = TaskSampler(TASK_COUNT, SEED+1, mode=method, history=HISTORY, target=TARGET,
                                  concentration=KAPPA, temperature=TEMPERATURE, epsilon=EPS)
        self.generator = torch.Generator(device=device).manual_seed(SEED+1)
        self.logits = torch.full((TASK_COUNT,), self.logit(.5), dtype=torch.float64, device=device)

    @staticmethod
    def logit(rate):
        weight = (rate+EPS)**(KAPPA*TARGET) * (1-rate+EPS)**(KAPPA*(1-TARGET))
        return math.log(max(weight, EPS)+EPS) / TEMPERATURE

    def track(self, ids):
        for task_id in ids:
            self.vendor.sample_counts[task_id] += 1

    def record(self, ids, success):
        self.vendor.record(ids,success)
        if self.method == "sgs":
            changed = sorted(set(ids))
            values = [self.logit(sum(self.vendor.outcomes[i])/len(self.vendor.outcomes[i])) for i in changed]
            ids = self.torch.tensor(changed, dtype=self.torch.long, device=self.device)
            self.logits[ids] = self.torch.tensor(values, dtype=self.logits.dtype, device=self.device)

    def sample(self, count):
        torch = self.torch
        if self.method == "uniform":
            ids = torch.randint(TASK_COUNT,(count,),generator=self.generator,device=self.device).tolist()
            probabilities = [1./TASK_COUNT]*count
        else:
            weights = self.logits.softmax(dim=0)
            selected = torch.multinomial(weights,count,replacement=True,generator=self.generator)
            ids, probabilities = selected.tolist(), weights[selected].tolist()
        self.track(ids)
        return ids, probabilities

    def audit(self):
        history = self.vendor
        p = self.logits.softmax(dim=0) if self.method=="sgs" else None
        return dict(natural_count=sum(history.completed_counts), successes=sum(history.success_counts),
                    stored_labels=sum(map(len,history.outcomes)), configurations_seen=sum(bool(x) for x in history.outcomes),
                    distinct_probabilities=(int(p.unique().numel()) if p is not None else 1),
                    probability_min=(float(p.min()) if p is not None else 1./TASK_COUNT),
                    probability_max=(float(p.max()) if p is not None else 1./TASK_COUNT),
                    max_history=max(map(len,history.outcomes)))

    def state(self):
        return dict(history=self.vendor.state_dict(),generator=self.generator.get_state(),logits=self.logits.clone())

    def load(self, state):
        self.vendor.load_state_dict(state["history"])
        self.generator.set_state(state["generator"].cpu())
        self.logits.copy_(state["logits"])


def verify_table_math(torch, device):
    from goalreach.vendor_sampler import TaskSampler

    test = TaskSampler(7, 31, mode="sgs", history=HISTORY, target=TARGET,
                       concentration=KAPPA, temperature=TEMPERATURE, epsilon=EPS)
    test.record([2]*103 + [0,0,2,3], [False]*103 + [True,False,True,True])
    expected = torch.as_tensor(test.probabilities(), dtype=torch.float64, device=device)
    observed = torch.tensor([Sampling.logit(x) for x in test.rates()],
                            dtype=torch.float64, device=device).softmax(dim=0)
    error = (expected-observed).abs().max().item()
    assert error < 1e-14 and len(test.outcomes[2])==HISTORY
    return dict(reference_max_probability_error=error, repeated_task_history_length=len(test.outcomes[2]))


def obs(world):
    from tensordict import TensorDict

    readings = world.observations()
    if not world.torch.isfinite(readings).all():
        raise RuntimeError("Non-finite observation")
    return TensorDict({"policy":readings}, batch_size=[world.n], device=world.device)


def make_algorithm(world, readings):
    from isaaclab_rl.rsl_rl import (RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg,
                                   handle_deprecated_rsl_rl_cfg)
    from rsl_rl.algorithms import PPO

    network = dict(hidden_dims=[512,256,128], activation="elu", obs_normalization=False)
    actor = RslRlMLPModelCfg(**network,distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.))
    critic = RslRlMLPModelCfg(**network)
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.,use_clipped_value_loss=True,clip_param=.2,
        entropy_coef=.005,num_learning_epochs=5,num_mini_batches=4,learning_rate=.001,schedule="adaptive",
        gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.)
    native = RslRlOnPolicyRunnerCfg(actor=actor,critic=critic,algorithm=algorithm,
        obs_groups={"actor":["policy"],"critic":["policy"]},num_steps_per_env=STEPS,max_iterations=RUN_UPDATES,
        save_interval=RUN_UPDATES,experiment_name="sgs_anymald_smoke")
    config = handle_deprecated_rsl_rl_cfg(native,version("rsl-rl-lib")).to_dict()
    config["multi_gpu"] = None
    return PPO.construct_algorithm(readings,world,config,world.device)


def replace_tasks(world, mapping):
    placed = world.reset_tasks(mapping)
    world.sim.forward()
    return placed, obs(world)


def correct_timeout(algorithm, final_obs, result, reward):
    truncated = result==4
    targets = algorithm.critic(final_obs).detach().squeeze(-1)
    adjusted = reward + algorithm.gamma * targets * truncated
    assert bool((adjusted[~truncated]==reward[~truncated]).all())
    return adjusted, targets


def timeout_control(world, algorithm, initial):
    torch = world.torch
    replace_tasks(world, dict(enumerate(initial)))
    # The first slot uses supported Gap terrain and a distinct physical goal.
    world.steps[0]=nav.MAX_STEPS-1
    actions = torch.zeros_like(world.last_action)
    result, signals = world.step(actions)
    assert int(result[0])==4
    final = obs(world)
    adjusted, values = correct_timeout(algorithm,final,result,signals["reward"])
    actual = float(adjusted[0]-signals["reward"][0])
    expected = float(values[0])*algorithm.gamma
    assert abs(actual-expected) < 1e-6
    return dict(task_id=initial[0],label=LABELS[4],age=int(world.steps[0]),rollout_history_written=False,
                raw_reward=float(signals["reward"][0]),terminal_value=float(values[0]),bootstrap=actual,
                gamma=algorithm.gamma,non_timeout_reward_unchanged=True)


class Arm:
    def __init__(self, world, method, out, initial):
        self.world, self.out, self.method = world, out, method
        self.out.mkdir()
        self.sampling = Sampling(world.torch,method,world.device)
        self.initial_ids = list(initial)
        self.natural, self.cross, self.steps, self.iteration = 0, 0, 0, 0
        self.outcomes = {name:Counter() for name in TERRAIN_NAMES}
        self.timeout_seen = 0
        self.changed_slots = 0

    def setup(self):
        world = self.world
        torch = world.torch
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        _, self.obs = replace_tasks(world,dict(enumerate(self.initial_ids)))
        self.sampling.track(self.initial_ids)
        algorithm = make_algorithm(world,self.obs)
        self.algorithm = algorithm
        before_cpu, before_cuda = torch.get_rng_state(), torch.cuda.get_rng_state(world.device)
        with torch.no_grad():
            control = timeout_control(world,algorithm,self.initial_ids)
        torch.set_rng_state(before_cpu)
        torch.cuda.set_rng_state(before_cuda,world.device)
        _, self.obs = replace_tasks(world,dict(enumerate(self.initial_ids)))
        initial_hash = fingerprint(algorithm.actor.state_dict())
        save(self.out/"initialization.json",dict(at=now(),method=self.method,actor_sha256=initial_hash,
            critic_sha256=fingerprint(algorithm.critic.state_dict()),initial_tasks=self.initial_ids,
            timeout_control=control,initial_terrain_counts=dict(Counter(TERRAIN_NAMES[decode(x)[0]]
                                                                        for x in self.initial_ids))))
        return initial_hash

    def update(self):
        world, alg = self.world, self.algorithm
        torch = world.torch
        device = world.device
        stage_start = time.perf_counter()
        starts = self.steps
        pre_hash = fingerprint(alg.actor.state_dict())
        alg.train_mode()
        rewards = []
        with torch.no_grad():
            for step in range(STEPS):
                action = alg.act(self.obs)
                assert torch.isfinite(action).all()
                result, values = world.step(action, training_iteration=self.iteration)
                assert torch.equal(world.last_action, action)
                rewards.append(float(values["reward"].mean()))
                final = obs(world)
                done = result>0
                ended = done.nonzero().flatten().tolist()
                reward, final_values = correct_timeout(alg,final,result,values["reward"])
                if ended:
                    old_ids = [world.ids[i] for i in ended]
                    labels = [LABELS[i] for i in result[ended].tolist()]
                    is_success = [label=="success" for label in labels]
                    self.sampling.record(old_ids,is_success)
                    new_ids, probability = self.sampling.sample(len(ended))
                    prior_pose = world.robot.root_physx_view.get_root_transforms().clone()
                    placed, next_readings = replace_tasks(world,dict(zip(ended,new_ids,strict=True)))
                    assert torch.equal(prior_pose[~done],world.robot.root_physx_view.get_root_transforms()[~done])
                    assert torch.count_nonzero(next_readings["policy"][ended,33:45])==0
                    assert torch.equal(final["policy"][~done],next_readings["policy"][~done])
                    self.changed_slots += sum(old!=new for old,new in zip(old_ids,new_ids,strict=True))
                    for j,(slot,old,new,label) in enumerate(zip(ended,old_ids,new_ids,labels,strict=True)):
                        a,b = decode(old)[:3],decode(new)[:3]
                        self.outcomes[TERRAIN_NAMES[a[0]]][label] += 1
                        self.natural += 1
                        self.timeout_seen += label=="timeout"
                        if a!=b:
                            self.cross += 1
                            if not (self.out/"first_natural_terrain_switch.json").is_file():
                                save(self.out/"first_natural_terrain_switch.json",dict(at=now(),slot=slot,
                                    update=self.iteration+1,control_step=step+1,old_id=old,new_id=new,
                                    old_geometry=a,new_geometry=b,natural_label=label,
                                    draw_probability=probability[j],native_spawn=placed[j].tolist(),
                                    final_observation_sha256=fingerprint(final["policy"][slot]),
                                    initial_observation_sha256=fingerprint(next_readings["policy"][slot]),
                                    final_observation_value=float(final_values[slot]),next_age=int(world.steps[slot])))
                else:
                    next_readings = final
                alg.process_env_step(next_readings,reward,done,{})
                assert torch.equal(alg.storage.rewards[step,:,0],reward)
                self.obs = next_readings
                self.steps += world.n
            alg.compute_returns(self.obs)
            torch.cuda.synchronize(device)
        collect = time.perf_counter()-stage_start
        count = [0]

        def count_step(*_):
            count[0] += 1

        hook = alg.optimizer.register_step_post_hook(count_step)
        begin = time.perf_counter()
        try:
            loss = alg.update()
            torch.cuda.synchronize(device)
        finally:
            hook.remove()
        optimized = time.perf_counter()-begin
        assert count[0]==20 and all(math.isfinite(float(v)) for v in loss.values())
        assert all(torch.isfinite(parameter).all() for parameter in (*alg.actor.parameters(), *alg.critic.parameters()))
        self.iteration += 1
        post_hash = fingerprint(alg.actor.state_dict())
        assert pre_hash!=post_hash and self.steps-starts==world.n*STEPS
        row = dict(at=now(),method=self.method,update=self.iteration,interactions=self.steps-starts,
            total_interactions=self.steps,optimizer_calls=count[0],before_actor_sha256=pre_hash,
            after_actor_sha256=post_hash,collect_s=collect,optimize_s=optimized,loss=loss,mean_reward=sum(rewards)/len(rewards),
            natural_total=self.natural,cross_terrain_total=self.cross,natural_timeouts=self.timeout_seen)
        append(self.out/"updates.jsonl",row)
        if self.iteration==1:
            save(self.out/"first_ppo.json",row)
        return row

    def evaluate(self, fixed):
        torch, world, algorithm = self.world.torch, self.world, self.algorithm
        frozen = fingerprint((algorithm.save(),self.sampling.state()))
        count = len(fixed)
        padded = fixed + [fixed[0]]*(world.n-count)
        _, readings = replace_tasks(world,dict(enumerate(padded)))
        active = torch.ones(world.n,dtype=torch.bool,device=world.device)
        active[count:] = False
        findings = {}
        started = time.perf_counter()
        algorithm.eval_mode()
        with torch.no_grad():
            for tick in range(nav.MAX_STEPS):
                action = algorithm.actor(readings)
                assert torch.isfinite(action).all()
                result, values = world.step(action,training_iteration=self.iteration)
                indices = (active & (result>0)).nonzero().flatten().tolist()
                for slot in indices:
                    findings[slot] = world.row(slot,result,values)
                active[result>0] = False
                if len(findings)==count:
                    break
                readings = obs(world)
        assert len(findings)==count and frozen==fingerprint((algorithm.save(),self.sampling.state()))
        report = {name:Counter() for name in TERRAIN_NAMES}
        for record in findings.values():
            report[record["terrain"]][record["label"]] += 1
        result = dict(at=now(),fixed_ids=fixed,seed=EVAL_SEED,valid=count,seconds=time.perf_counter()-started,
            successes=sum(rec["label"]=="success" for rec in findings.values()),
            by_terrain={name:dict(value) for name,value in report.items()},training_state_sha256=frozen,
            training_history_unchanged=True,rows=list(findings.values()))
        save(self.out/"fixed_evaluation.json",result)
        return result

    def save_and_restore(self, fixed):
        torch, world = self.world.torch, self.world
        continuation_ids, probabilities = self.sampling.sample(world.n)
        checkpoint = self.out/"checkpoint.pt"
        before = self.algorithm.save()
        checkpoint_state = dict(version=EDITION,geometry=GEOMETRY_VERSION,iteration=self.iteration,arm=self.method,
            ppo=before,sampler=self.sampling.state(),learning_rate=self.algorithm.learning_rate,
            rng_cpu=torch.get_rng_state(),rng_gpu=torch.cuda.get_rng_state(world.device),
            continuation_ids=continuation_ids,continuation_probabilities=probabilities,boundary="fresh_episodes")
        torch.save(checkpoint_state,checkpoint)
        saved = fingerprint((before,checkpoint_state["sampler"]))
        # This separate fixed panel runs without training's sampler or transitions.
        evaluation = self.evaluate(fixed)
        _, self.obs = replace_tasks(world,dict(enumerate(continuation_ids)))
        restored = make_algorithm(world,self.obs)
        data = torch.load(checkpoint,map_location=world.device,weights_only=True)
        assert data["version"]==EDITION and data["geometry"]==GEOMETRY_VERSION and data["iteration"]==self.iteration
        restored.load(data["ppo"],load_cfg=None,strict=True)
        restored.learning_rate=data["learning_rate"]
        self.sampling = Sampling(torch,self.method,world.device)
        self.sampling.load(data["sampler"])
        assert saved==fingerprint((restored.save(),self.sampling.state()))
        torch.set_rng_state(data["rng_cpu"].cpu())
        torch.cuda.set_rng_state(data["rng_gpu"].cpu(),world.device)
        self.algorithm = restored
        update = self.update()
        with checkpoint.open("rb") as handle:
            from hashlib import file_digest
            checkpoint_sha = file_digest(handle,"sha256").hexdigest()
        continuation = dict(at=now(),checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_sha,
            checkpoint_bytes=checkpoint.stat().st_size,checkpoint_iteration=data["iteration"],
            restart_boundary="fresh_episodes",open_episodes_excluded=world.n,
            learner_and_sampler_exact=True,resumed_update=update)
        save(self.out/"restored_ppo.json",continuation)
        return evaluation, continuation


def initial_tasks(num_envs):
    rng = random.Random(SEED)
    layouts = [(slot%len(TERRAIN_NAMES),rng.randrange(2),rng.randrange(10)) for slot in range(num_envs)]
    layouts[0]=(0,0,0)
    return [encode(*layout,rng.randrange(20),rng.randrange(20)) for layout in layouts]


def execute(output, num_envs):
    import torch
    from goalreach import vendor_sampler

    bank = [(kind,seed,level) for kind in range(len(TERRAIN_NAMES))
            for seed in range(TERRAIN_SEEDS) for level in range(LEVELS)]
    initial = initial_tasks(num_envs)
    layouts = [decode(task_id)[:3] for task_id in initial]
    eval_rng = random.Random(EVAL_SEED)
    fixed = [encode(kind,eval_rng.randrange(2),eval_rng.randrange(10),
                    eval_rng.randrange(20),eval_rng.randrange(20))
             for kind in range(len(TERRAIN_NAMES)) for _ in range(2)]
    status = dict(at=now(),version=EDITION,geometry=GEOMETRY_VERSION,state="loading_terrain_atlas",
                  gpu=torch.cuda.get_device_name(0),num_envs=num_envs,steps_per_update=STEPS,
                  interactions_per_update=num_envs*STEPS,table_math=verify_table_math(torch,"cuda:0"))
    save(output/"ppo_progress.json",status)
    world = GoalScene(layouts,False,tile_bank=bank)
    world.num_envs, world.num_actions = world.n, 12
    arms, hashes = {}, []
    for method in ("uniform","sgs"):
        stage = Arm(world,method,output/method,initial)
        status.update(at=now(),state="starting_arm",arm=method,completed_arms=list(arms))
        save(output/"ppo_progress.json",status)
        hashes.append(stage.setup())
        for _ in range(RUN_UPDATES):
            row = stage.update()
            status.update(at=now(),state="training",last_update=row)
            save(output/"ppo_progress.json",status)
        assert stage.natural and stage.cross
        status.update(at=now(),state="fixed_evaluation_and_resume")
        save(output/"ppo_progress.json",status)
        panel, restored = stage.save_and_restore(fixed)
        actual = stage.sampling.audit()
        assert actual["natural_count"]==stage.natural and actual["successes"]==sum(
            value["success"] for value in stage.outcomes.values())
        if method=="sgs":
            assert actual["distinct_probabilities"]>1
        arms[method] = dict(ppo_updates=stage.iteration,interactions=stage.steps,natural=stage.natural,
                           natural_timeouts=stage.timeout_seen,cross=stage.cross,training=stage.outcomes,
                           sampling=actual,eval_valid=panel["valid"],eval_successes=panel["successes"],
                           eval_by_terrain=panel["by_terrain"],restore=restored)
        save(output/"arms_progress.json", arms)
    assert len(set(hashes))==1
    status.update(at=now(),state="complete",completed_arms=list(arms),ppo_updates=sum(a["ppo_updates"] for a in arms.values()))
    save(output/"ppo_progress.json",status)
    save(output/"scene_receipt.json",dict(status,initial_actor_sha256=hashes[0],arms=arms,
        library=version("rsl-rl-lib"),vendor_source_sha256=sha256(Path(vendor_sampler.__file__).read_bytes()).hexdigest(),
        sampler_parameters=dict(target=TARGET,kappa=KAPPA,epsilon=EPS,temperature=TEMPERATURE,
                                history=HISTORY,empty_history_prior=.5),evaluation_ids=fixed,
        scene_geometry=world.geometry))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--envs",type=int,default=64)
    parser.add_argument("--experience",required=True)
    parser.add_argument("--kit-args",required=True)
    args = parser.parse_args()
    if args.envs < 2*len(TERRAIN_NAMES):
        parser.error("fixed panels require at least 26 environments")
    from isaaclab.app import AppLauncher
    from tools.p3s2_frozen_protocol import launcher_class

    app = launcher_class(AppLauncher,1)(headless=True,device="cuda:0",enable_cameras=False,
        livestream=0,experience=args.experience,kit_args=args.kit_args).app
    try:
        execute(args.output,args.envs)
    except BaseException:
        (args.output/"scene_error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(),flush=True)
        raise
    finally:
        app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__ == "__main__":
    main()
