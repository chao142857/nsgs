"""Live state assignment, terminal capture and observations; targets only change on reset."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from .core import EpisodeState, Sampler, body_vectors, append_rows, failure_reward
from .kinematics import Arm


class TaskRuntime:
    def __init__(self, env):
        self.env = env
        p = Path(env.cfg.goalreach['pool'])
        self.pool = dict(np.load(p, allow_pickle=False))
        self.pool_sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
        self.targets = torch.as_tensor(self.pool['target'], device=env.device)
        self.deltas = torch.as_tensor(self.pool['initial_joint_delta'], device=env.device)
        arm=Arm(Path(__file__).resolve().parents[1]/'reference/native_kinematics.json')
        self.initial_ee=torch.as_tensor(self.pool['initial_ee'] if 'initial_ee' in self.pool else
            arm.fk(self.pool['initial_joint_delta'][:,-6:]),device=env.device,dtype=torch.float32)
        self.state = EpisodeState(env.num_envs, env.device, env.step_dt,
                                  env.cfg.goalreach.get('success_criterion', 'hold_and_speed'))
        self.sampler = Sampler(self.pool['region'], env.cfg.seed,
            mode='external' if env.cfg.goalreach['sampler']=='manual' else env.cfg.goalreach['sampler'], region=env.cfg.goalreach['region'])
        self.curriculum=None
        if env.cfg.goalreach['sampler']=='manual':
            from .curriculum import ManualCurriculum
            self.curriculum=ManualCurriculum(self.pool['region'],**env.cfg.goalreach.get('curriculum',{}))
            self.sampler.set_probabilities(self.curriculum.probabilities())
        if env.cfg.goalreach.get('probability'):
            self.sampler.set_probabilities(np.load(env.cfg.goalreach['probability']))
        robot = env.scene['robot']
        self.ee = robot.find_bodies('end_effector')[0][0]
        self.action_ids = robot.find_joints(env.cfg.actions.joint_pos.joint_names, preserve_order=True)[0]
        if 'action_joint_names' in self.pool:
            assert self.pool['action_joint_names'].tolist()==[robot.joint_names[j] for j in self.action_ids]
            self.initial_root_poses=torch.as_tensor(self.pool['initial_root_pose'],device=env.device)
            self.initial_joint_positions=torch.as_tensor(self.pool['initial_joint_position'],device=env.device)
        self.base = env.scene.sensors['contact_forces'].body_names.index('base')
        self.nonfoot = [i for i,n in enumerate(env.scene.sensors['contact_forces'].body_names) if not n.endswith('_foot')]
        self.last_tick = -1
        self.forced_ids = None
        self.last_rows = []
        self.out = Path(env.cfg.goalreach['out'])
        self.out.mkdir(parents=True, exist_ok=True)
        self.completed = 0
        self.reward_sums = {}
        self.reward_episodes = 0
        from .training_metrics import TrainingWindow
        self.training_window=TrainingWindow()
        self.first_training_success=None
        from .episode_log import EpisodeLog, SamplingDiagnostics
        self.episode_log=EpisodeLog(self.out,self.pool_sha256) if env.cfg.goalreach.get('binary_log') and self.sampler.interface_mode!='evaluation' else None
        self.sampling_diagnostics=SamplingDiagnostics()
        self.assignment_metadata={}
        self.minimum_base_height=torch.full((env.num_envs,),float('inf'),device=env.device)

    def end_update(self,update):
        if self.curriculum:
            event=self.curriculum.end_update(update)
            if event:append_rows(self.out/'curriculum_events.jsonl',[event])
            self.sampler.set_probabilities(self.curriculum.probabilities())
        if update%50==0:
            append_rows(self.out/'sampling_diagnostics.jsonl',[self.sampling_diagnostics.export(self.sampler,update)])
            from .episode_log import SamplingDiagnostics
            self.sampling_diagnostics=SamplingDiagnostics()
        if self.episode_log:self.episode_log.flush(update)


    def assign(self, ids):
        capture=self.env.cfg.goalreach.get('p2a_capture',False)
        if capture:
            probabilities=self.sampler.probabilities()
            visits=self.sampler.sample_counts.copy()
        if self.forced_ids is not None:
            tasks = self.forced_ids
            self.forced_ids = None
            assert len(tasks) == len(ids)
        else:
            tasks = self.sampler.sample(len(ids))
        tasks = torch.as_tensor(tasks, device=self.env.device, dtype=torch.long)
        # Initial yaw is fixed at zero. z is above the flat ground, not above root.
        target = self.targets[tasks] + self.env.scene.env_origins[ids]
        self.state.reset(ids, tasks, target)
        if capture:
            for e,t in zip(ids.cpu().tolist(),tasks.cpu().tolist()):
                rate=float(self.sampler.cached_rates[t])
                self.assignment_metadata[e]=dict(sample_update=self.env.cfg.goalreach.get('training_update',1),
                    sample_probability=probabilities[t],sample_success_rate=rate,sample_visits=visits[t],
                    curriculum_stage=self.curriculum.stage if self.curriculum else -1)
                visits[t]+=1
                self.sampling_diagnostics.add(int(self.pool['region'][t]),rate,self.sampler.completed_counts[t])
            self.minimum_base_height[ids]=self.initial_root_poses[tasks,2]
        initial_distance=(self.targets[tasks]-self.initial_ee[tasks]).norm(dim=-1)
        self.state.distance[ids]=self.state.minimum[ids]=self.state.previous[ids]=initial_distance
        if self.env.cfg.goalreach.get('legacy_jsonl',True):append_rows(self.out/'assignments.jsonl', [dict(env_id=e, task_id=t, generation=g,
            target_world_m=v, pool_sha256=self.pool_sha256) for e,t,g,v in zip(
            ids.cpu().tolist(),tasks.cpu().tolist(),self.state.generation[ids].cpu().tolist(),target.cpu().tolist())])
        return tasks

    def tick(self):
        if self.last_tick == self.env.common_step_counter:
            return
        self.last_tick = self.env.common_step_counter
        robot = self.env.scene['robot'].data
        force = self.env.scene.sensors['contact_forces'].data.net_forces_w_history.norm(dim=-1)
        base = force[:,:,self.base].amax(1) > .5
        tipped = torch.acos((-robot.projected_gravity_b[:,2]).clamp(-1,1)) > 1.
        # Same >5 N and strictly >0.5s grace rule as the October task suite.
        nonfoot = (force[:,:,self.nonfoot].amax(dim=(1,2)) > 5.) & ((self.state.steps+1)*self.env.step_dt > .5)
        # body_lin_vel_w is a deprecated COM-velocity alias in this Isaac build.
        # Distance and speed must refer to the same end-effector link origin.
        if self.env.cfg.goalreach.get('p2a_capture'):
            height=robot.root_pos_w[:,2]-self.env.scene.env_origins[:,2]
            self.minimum_base_height=torch.minimum(self.minimum_base_height,height)
        self.state.advance(robot.body_pos_w[:,self.ee], robot.body_link_lin_vel_w[:,self.ee], base,tipped,nonfoot)

    def finish(self, ids):
        done = ids[self.state.reason[ids] != 0]
        if not len(done):
            return
        rows = self.state.rows(done)
        if self.env.cfg.goalreach.get('training_metrics'):
            robot=self.env.scene['robot'].data
            origin_ee=self.initial_ee[self.state.task[done]]+self.env.scene.env_origins[done]
            displacement=(robot.body_pos_w[done,self.ee]-origin_ee).norm(dim=-1).cpu().tolist()
            initial_distance=(self.targets[self.state.task[done]]-self.initial_ee[self.state.task[done]]).norm(dim=-1).cpu().tolist()
            for row,move,d0 in zip(rows,displacement,initial_distance):
                row.update(ee_displacement_m=move,initial_to_terminal_progress_m=d0-row['terminal_distance_m'])
        if self.env.cfg.goalreach.get('diagnostic_capture'):
            from tools.diagnostic_helpers import terminal_details,conditions
            extra=terminal_details(self.env,done);physics=conditions(self.env,done)
            for row,details,condition in zip(rows,extra,physics):row.update(details,physics_condition=condition)
        for row, i in zip(rows, done.cpu().tolist()):
            row.update(env_id=i, pool_sha256=self.pool_sha256,
                       region=int(self.pool['region'][row['task_id']]),
                       height_stratum=int(self.pool['height_stratum'][row['task_id']]),
                       target_height_m=float(self.pool['height'][row['task_id']]))
            row['near_distance_band']=int(self.pool['near_distance_band'][row['task_id']]) if 'near_distance_band' in self.pool else -1
        if self.env.cfg.goalreach.get('p2a_capture'):
            robot=self.env.scene['robot'].data
            task_ids = [row['task_id'] for row in rows]
            q = robot.root_quat_w[done]
            w, x, y, z = q.unbind(-1)
            yaw = torch.atan2(2*(w*z+x*y), 1-2*(y**2+z**2))
            target_azimuth = torch.as_tensor(self.pool['azimuth'][task_ids], device=self.env.device)
            angle = yaw-target_azimuth
            values = torch.stack((robot.root_pos_w[done,2]-self.env.scene.env_origins[done,2],
                                  self.minimum_base_height[done],
                                  torch.abs(torch.atan2(torch.sin(angle),torch.cos(angle)))), dim=-1).cpu().tolist()
            cells = self.pool.get('cell_id')
            sectors = self.pool.get('azimuth_sector')
            for row, i, (height, minimum, heading) in zip(rows, done.cpu().tolist(), values):
                task = row['task_id']
                row.update(self.assignment_metadata[i])
                row.update(end_update=self.env.cfg.goalreach.get('training_update',1),
                    terminal_base_height_m=height, minimum_base_height_m=minimum,
                    terminal_heading_error_rad=heading,
                    cell_id=int(cells[task]) if cells is not None else -1,
                    azimuth_sector=int(sectors[task]) if sectors is not None else -1)
                if self.env.cfg.goalreach.get('training_metrics'):
                    if self.episode_log:self.episode_log.add(row)
                    if self.curriculum:self.curriculum.record(row['end_update'],row['region'],row['success'])
        self.sampler.record([r['task_id'] for r in rows], [r['success'] for r in rows])
        if self.env.cfg.goalreach.get('legacy_jsonl',True):append_rows(self.out/'terminal_feedback.jsonl', rows)
        self.last_rows.extend(rows)
        self.completed += len(rows)
        reward_batch={}
        for name, values in self.env.reward_manager._episode_sums.items():
            value=float(values[done].sum());reward_batch[name]=value
            self.reward_sums[name] = self.reward_sums.get(name,0.) + value
        if self.env.cfg.goalreach.get('training_metrics'):
            self.training_window.add(rows,reward_batch)
            if self.first_training_success is None and any(r['success'] for r in rows):
                from .core import write_json
                self.first_training_success=dict(update=self.env.cfg.goalreach.get('training_update'),
                    action_std=self.env.cfg.goalreach.get('sampling_action_std'),
                    control_tick=self.env.common_step_counter,example=next(r for r in rows if r['success']))
                write_json(self.out/'first_training_success.json',self.first_training_success)
        self.reward_episodes += len(done)
        self.state.reason[done] = 0
        self.state.task[done] = -1


def runtime(env):
    if not hasattr(env, 'goal_runtime'):
        env.goal_runtime = TaskRuntime(env)
    return env.goal_runtime


def reset_robot(env, env_ids):
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if not len(ids): return
    rt = runtime(env)
    tasks = rt.assign(ids)
    robot = env.scene['robot']
    root = robot.data.default_root_state[ids].clone()
    root[:,:3] += env.scene.env_origins[ids]
    root[:,3:7] = torch.tensor([1.,0.,0.,0.],device=env.device)
    root[:,7:13] = 0.
    q = robot.data.default_joint_pos[ids].clone()
    if 'initial_root_pose' in rt.pool:
        root[:,:7]=rt.initial_root_poses[tasks]
        root[:,:3]+=env.scene.env_origins[ids]
        q[:,rt.action_ids]=rt.initial_joint_positions[tasks]
    elif env.cfg.goalreach.get('diagnostic_initial_reference'):
        # Diagnostic cold-start only. The native default_joint_pos (the zero
        # action's PD target) is deliberately NOT changed to the measured q.
        reference=env.cfg.goalreach['diagnostic_initial_reference']
        assert reference['action_joint_names']==[robot.joint_names[j] for j in rt.action_ids]
        root[:,:7]=torch.as_tensor(reference['root_pose'],device=env.device)
        root[:,:3]+=env.scene.env_origins[ids]
        q[:,rt.action_ids]=torch.as_tensor(reference['joint_position'],device=env.device)
    if 'initial_joint_position' not in rt.pool:
        q[:,rt.action_ids] += rt.deltas[tasks]
    robot.write_root_pose_to_sim(root[:,:7],env_ids=ids)
    robot.write_root_velocity_to_sim(root[:,7:13],env_ids=ids)
    robot.write_joint_state_to_sim(q, torch.zeros_like(q),env_ids=ids)


def terminal(env):
    rt = runtime(env);rt.tick()
    return rt.state.reason != 0


def target_body(env):
    rt = runtime(env);r=env.scene['robot'].data
    return body_vectors(r.root_quat_w,r.root_pos_w,r.body_pos_w[:,rt.ee],rt.state.target)[0]


def error_body(env):
    rt = runtime(env);r=env.scene['robot'].data
    return body_vectors(r.root_quat_w,r.root_pos_w,r.body_pos_w[:,rt.ee],rt.state.target)[1]


def remaining_time(env):
    return ((500-runtime(env).state.steps).clamp(0,500).float()/500).unsqueeze(-1)


def terminal_success(env):
    return (runtime(env).state.reason == 1).float()/env.step_dt


def terminal_failure(env):
    # Reward manager still applies the original -100 weight and dt.
    return failure_reward(runtime(env).state.reason,
        env.cfg.goalreach.get('timeout_penalty',-100.))/(-100.*env.step_dt)


def progress(env):
    return runtime(env).state.progress/env.step_dt

