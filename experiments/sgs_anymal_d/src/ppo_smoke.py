"""Shared frozen PPO construction, observation adapter and checkpoint/log helpers."""
from hashlib import sha256
from importlib.metadata import version
import json

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


def obs(world):
    from tensordict import TensorDict

    readings = world.observations()
    if not world.torch.isfinite(readings).all():
        raise RuntimeError("Non-finite observation")
    return TensorDict({"policy":readings}, batch_size=[world.n], device=world.device)


def make_algorithm(world, readings, total_updates=500):
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
        obs_groups={"actor":["policy"],"critic":["policy"]},num_steps_per_env=STEPS,max_iterations=total_updates,
        save_interval=total_updates,experiment_name="sgs_anymal_d_figure7_pilot")
    config = handle_deprecated_rsl_rl_cfg(native,version("rsl-rl-lib")).to_dict()
    config["multi_gpu"] = None
    return PPO.construct_algorithm(readings,world,config,world.device)


