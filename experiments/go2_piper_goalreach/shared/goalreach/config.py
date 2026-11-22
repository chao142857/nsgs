"""Derive the native WBC configuration without altering its registry or source."""
from pathlib import Path
import math


def build_cfg(pool, out, num_envs=64, seed=42, reward='R0', sampler='uniform', region=None, device='cuda:0', probability=None, timeout_penalty=-50.):
    from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.go2_piper.wbc_env_cfg import Go2PiperWBCEnvCfg
    from . import mdp
    if reward not in ('R0','R1'):raise ValueError(reward)
    if timeout_penalty not in (-100.,-50.):raise ValueError(timeout_penalty)
    cfg = Go2PiperWBCEnvCfg()
    cfg.scene.num_envs = num_envs
    cfg.scene.env_spacing = 8.
    cfg.seed, cfg.sim.device = seed, device
    if device=='cpu':
        # Fabric is a GPU-backed transport; USD updates allow a no-GPU CPU check.
        cfg.sim.use_fabric=False
    cfg.episode_length_s = 10.
    cfg.is_finite_horizon = True
    cfg.commands.ee_pose = cfg.commands.base_velocity = None
    for name in ('lin_vel_cmd_levels','ang_vel_cmd_levels','pos_cmd_levels'):
        setattr(cfg.curriculum,name,None)
    for group in (cfg.observations.policy,cfg.observations.critic):
        group.velocity_commands = group.pos_commands = None
        group.target_body = ObservationTermCfg(func=mdp.target_body)
        group.error_body = ObservationTermCfg(func=mdp.error_body)
        group.remaining_time = ObservationTermCfg(func=mdp.remaining_time)
    cfg.events.reset_base = EventTermCfg(func=mdp.reset_robot,mode='reset')
    cfg.events.reset_robot_joints = None
    # The native startup material/mass/COM/inertia and reset gain randomization remain.
    for name in vars(cfg.terminations):
        if not name.startswith('_'):setattr(cfg.terminations,name,None)
    cfg.terminations.goalreach_terminal = TerminationTermCfg(func=mdp.terminal,time_out=False)
    remove = ('end_effector_position_tracking_exp','end_effector_orientation_tracking',
        'track_lin_vel_xy_exp','track_ang_vel_z_exp','track_base_height_exp','feet_air_time',
        'feet_long_air','air_time_variance','joint_mirror')
    for name in remove:setattr(cfg.rewards,name,None)
    for name in ('flat_orientation_l2','arm_deviation','joint_deviation','hip_deviation'):
        getattr(cfg.rewards,name).weight *= .1
    cfg.rewards.goal_success = RewardTermCfg(func=mdp.terminal_success,weight=1000.)
    cfg.rewards.goal_failure = RewardTermCfg(func=mdp.terminal_failure,weight=-100.)
    cfg.rewards.goal_progress = RewardTermCfg(func=mdp.progress,weight=1.) if reward=='R1' else None
    cfg.goalreach=dict(pool=str(Path(pool)),out=str(Path(out)),reward=reward,sampler=sampler,
        region=region,probability=probability,timeout_penalty=timeout_penalty,success_radius_m=.05,hold_steps=25,
        speed_max_m_s=.25,timeout_steps=500,observation='radial clamp at 1m; full body quaternion')
    return cfg


def make_env(cfg):
    from isaaclab.envs import ManagerBasedRLEnv
    from . import mdp
    class GoalReachEnv(ManagerBasedRLEnv):
        def step(self, action):
            if hasattr(self,'goal_runtime'):
                # Persistent feedback lives on disk. Keep only the current step
                # in memory, including during long formal training.
                self.goal_runtime.last_rows=[]
            return super().step(action)
        def _reset_idx(self, env_ids):
            if hasattr(self,'goal_runtime'):
                self.goal_runtime.finish(env_ids)
            super()._reset_idx(env_ids)
    return GoalReachEnv(cfg=cfg)


def agent_cfg(seed=42, initial_action_std=1., entropy_coef=.005):
    import importlib.metadata as metadata
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
    from LeggedManip_Lab.tasks.manager_based.leggedmanip_lab.config.go2_piper.agents.rsl_rl_ppo_cfg import Go2PiperWBCPPORunnerCfg
    cfg=handle_deprecated_rsl_rl_cfg(Go2PiperWBCPPORunnerCfg(),metadata.version('rsl-rl-lib'))
    cfg.seed=seed
    if not math.isfinite(initial_action_std) or initial_action_std <= 0:
        raise ValueError('Initial action std must be finite and positive')
    if not math.isfinite(entropy_coef) or entropy_coef < 0:
        raise ValueError('Entropy coefficient must be finite and nonnegative')
    cfg.actor.distribution_cfg.init_std = initial_action_std
    cfg.algorithm.entropy_coef = entropy_coef
    return cfg
