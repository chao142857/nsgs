"""Read-only acceptance evidence, shared with the instrumented 2A reference."""
import hashlib
import json

from goalreach.core import write_json


def array_digest(value):
    value = value.detach().cpu().contiguous().numpy()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def training_state_digest(runtime):
    state = dict(sampler=runtime.sampler.state_dict(),
                 curriculum=runtime.curriculum.state_dict() if runtime.curriculum else None)
    if getattr(runtime,'neural',None):
        state['neural'] = runtime.neural.digest()
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def write_initial(out, env, runtime, runner, observations, sampler_digest=training_state_digest):
    import random

    import numpy as np
    import torch

    robot = env.scene['robot'].data
    policy_states = {}
    for name in ('actor', 'critic'):
        policy = getattr(runner.alg, name)
        policy_states[name] = {key: array_digest(value) for key, value in policy.state_dict().items()}
    initial = json.loads((out/'initial_training_state.json').read_text())
    write_json(out/'validation_initial.json', dict(
        env_seed=int(env.cfg.seed), student_seed=initial['student_seed'],
        states=policy_states, observations={key:array_digest(value) for key, value in observations.items()},
        tasks=array_digest(runtime.state.task), robot_root=array_digest(robot.root_state_w),
        robot_joint_position=array_digest(robot.joint_pos), sampler=sampler_digest(runtime),
        torch_rng=array_digest(torch.get_rng_state()),
        cuda_rng=[array_digest(value) for value in torch.cuda.get_rng_state_all()],
        python_rng_sha256=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        numpy_rng_sha256=hashlib.sha256(
            np.random.get_state()[1].tobytes() + repr(np.random.get_state()[2:]).encode()).hexdigest()))


def trace_assignments(runtime_class):
    """Digest every training assignment in reset order, including censored episodes."""
    native_assign = runtime_class.assign
    digest = hashlib.sha256()
    count = 0
    batches = 0

    def assign(runtime, ids):
        nonlocal count, batches
        tasks = native_assign(runtime, ids)
        if runtime.sampler.interface_mode != 'evaluation':
            columns = (ids, tasks, runtime.state.generation[ids])
            for column in columns:
                digest.update(column.detach().cpu().contiguous().numpy().tobytes())
            count += len(ids)
            batches += 1
        return tasks

    runtime_class.assign = assign
    return lambda: dict(assignments=count, batches=batches, sha256=digest.hexdigest())
