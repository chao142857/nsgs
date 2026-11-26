"""Shared PPO construction, bounded heteroscedastic policy, privileged critic adapter, and fingerprint helpers."""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
import math


STEPS = 24


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


def _define_bounded_heteroscedastic():
    import torch
    from torch.distributions import Normal
    from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution

    class BoundedHeteroscedasticGaussianDistribution(HeteroscedasticGaussianDistribution):
        """State-dependent diagonal Gaussian with clamped log-std bounds (prevents global std collapse)."""

        def __init__(
            self,
            output_dim: int,
            init_std: float = 0.85,
            std_type: str = "log",
            std_bounds: tuple[float, float] | list[float] = (0.06, 1.35),
        ) -> None:
            super().__init__(output_dim=output_dim, init_std=float(init_std), std_type=std_type)
            lo, hi = float(std_bounds[0]), float(std_bounds[1])
            if not (0.0 < lo < hi):
                raise ValueError(f"Invalid std_bounds: {std_bounds}")
            self.std_min = lo
            self.std_max = hi
            self.log_min = math.log(lo)
            self.log_max = math.log(hi)

        def update(self, mlp_output: torch.Tensor) -> None:
            mean, raw = torch.unbind(mlp_output, dim=-2)
            if self.std_type == "log":
                std = torch.exp(raw.clamp(self.log_min, self.log_max))
            else:
                std = raw.clamp(self.std_min, self.std_max)
            self._distribution = Normal(mean, std)

    return BoundedHeteroscedasticGaussianDistribution


_BOUNDED_HETERO_CLS = None


def get_bounded_heteroscedastic_cls():
    global _BOUNDED_HETERO_CLS
    if _BOUNDED_HETERO_CLS is None:
        _BOUNDED_HETERO_CLS = _define_bounded_heteroscedastic()
    return _BOUNDED_HETERO_CLS


class BoundedHeteroscedasticProxy:
    """Callable factory resolved by rsl_rl.utils.resolve_callable('ppo_smoke:BoundedHeteroscedasticProxy')."""

    def __new__(cls, output_dim: int, **kwargs):
        real_cls = get_bounded_heteroscedastic_cls()
        return real_cls(output_dim, **kwargs)


def obs(world, *, clip_height: bool = False, include_critic: bool = False):
    from tensordict import TensorDict

    readings = world.observations(clip_height=True) if clip_height else world.observations()
    if not world.torch.isfinite(readings).all():
        raise RuntimeError("Non-finite observation")
    payload = {"policy": readings}
    if include_critic:
        c_obs = world.critic_observations(readings)
        if not world.torch.isfinite(c_obs).all():
            raise RuntimeError("Non-finite critic observation")
        payload["critic"] = c_obs
    return TensorDict(payload, batch_size=[world.n], device=world.device)


def make_algorithm(
    world,
    readings,
    total_updates: int = 500,
    *,
    gamma: float = 0.99,
    hygiene: bool = False,
    entropy_coef: float = 0.005,
    dist_mode: str = "gaussian",
    init_std: float = 1.0,
    std_bounds: tuple[float, float] = (0.06, 1.35),
    use_critic_priv: bool = False,
    use_clipped_value_loss: bool | None = None,
):
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
        handle_deprecated_rsl_rl_cfg,
    )
    from rsl_rl.algorithms import PPO

    network = dict(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=hygiene)
    actor = RslRlMLPModelCfg(**network, distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=float(init_std)))
    critic = RslRlMLPModelCfg(**network)
    clipped_vl = (not hygiene) if use_clipped_value_loss is None else bool(use_clipped_value_loss)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=clipped_vl,
        clip_param=0.2,
        entropy_coef=entropy_coef,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=0.001,
        schedule="adaptive",
        gamma=gamma,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    obs_groups = {"actor": ["policy"], "critic": ["critic" if use_critic_priv else "policy"]}
    native = RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=algorithm,
        obs_groups=obs_groups,
        num_steps_per_env=STEPS,
        max_iterations=total_updates,
        save_interval=total_updates,
        experiment_name="sgs_anymal_d_figure7_pilot",
    )
    config = handle_deprecated_rsl_rl_cfg(native, version("rsl-rl-lib")).to_dict()
    if dist_mode == "heteroscedastic":
        config["actor"]["distribution_cfg"] = {
            "class_name": "ppo_smoke:BoundedHeteroscedasticProxy",
            "init_std": float(init_std),
            "std_type": "log",
            "std_bounds": [float(std_bounds[0]), float(std_bounds[1])],
        }
    elif dist_mode != "gaussian":
        raise ValueError(f"Unsupported dist_mode: {dist_mode}")
    config["multi_gpu"] = None
    return PPO.construct_algorithm(readings, world, config, world.device)
