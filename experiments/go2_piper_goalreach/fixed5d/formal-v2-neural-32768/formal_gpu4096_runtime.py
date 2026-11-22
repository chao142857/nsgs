"""Record the native mode and contact evidence inside the formal training run."""
import os
from datetime import datetime
from pathlib import Path
from time import monotonic

from fixed5d_bench_runtime import Evidence, read_births


class FormalEvidence(Evidence):
    def step(self, actions):
        result = super().step(actions)
        if self.steps == 768:
            self.put(self.out / 'physical_passed.json', dict(
                at=datetime.now().astimezone().isoformat(), controls=self.steps,
                validated_through_rollout=32, **self.finish()))
        return result


def attest(args, out, rt, env, wrapped, runner, spec, put):
    import warp

    learner = str(next(runner.alg.actor.parameters()).device)
    scorer = str(rt.neural.estimator.device)
    assert env.num_envs == spec['num_envs'] and env.num_envs in (2048,4096,8192,16384,32768)
    assert learner == scorer == spec['learner_device'] == 'cuda:0'
    assert len(rt.pool['task_id']) == 1048576
    assert rt.neural.estimator.fit_steps == 0 and not runner.alg.optimizer.state
    birth = read_births(rt, env)
    evidence = FormalEvidence(rt, env, wrapped, True, True, True, 32)
    evidence.put, evidence.out = put, out
    identity = Path('/proc/self/stat').read_text().rsplit(') ', 1)[1].split()
    put(out / 'physical_entry.json', dict(
        at=datetime.now().astimezone().isoformat(), worker_namespace_pid=os.getpid(),
        worker_start_ticks=int(identity[19]), execution_release=spec['execution_release'],
        seed=spec['seed'], initial_actor_sha256=spec['actor_sha256'],
        weights_sha256=spec['initial_weights_sha256'], mode=evidence.mode, birth=birth,
        learner=learner, scorer=scorer, training_parallel_envs=env.num_envs,
        evaluation_parallel_envs=env.num_envs, steps_per_env_per_update=24,
        training_interactions_per_update=24 * env.num_envs, training_pool_rows=1048576,
        startup_s=monotonic()-args.formal_started, warp_cache=str(warp.config.kernel_cache_dir)))
