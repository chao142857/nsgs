"""Exercise natural feedback past the cold start and resume PPO after fixed evaluation."""
import json


def run_probe(args,out,rt,env,wrapped,runner,ema,spec):
    from spatial_train import evaluate, put

    budget=out/'ppo_budget.json'
    runner.learn(num_learning_iterations=24,init_at_random_ep_len=False)
    rt.episode_log.flush(24,final=True)
    before=json.loads(budget.read_bytes())
    labels=int(rt.sampler.completed_counts.sum())
    wins=int(rt.sampler.success_counts.sum())
    fit=rt.neural.estimator.fit_steps
    assert before['ppo_updates_completed']==before['scorer_updates_passed']==24
    assert before['optimizer_calls']==480 and labels>wins>0 and fit>0
    outcome=evaluate(env,wrapped,runner,ema,args.compact_panel,500,out,rt,spec)
    assert outcome['rows']==504 and json.loads(budget.read_bytes())==before
    assert labels==int(rt.sampler.completed_counts.sum()) and fit==rt.neural.estimator.fit_steps
    runner.current_learning_iteration=24
    runner.learn(num_learning_iterations=1,init_at_random_ep_len=False)
    rt.episode_log.flush(25,final=True)
    after=json.loads(budget.read_bytes())
    assert after['ppo_updates_completed']==after['scorer_updates_passed']==25
    assert after['optimizer_calls']==500
    put(out/'passed.json',dict(passed=True,engineering_only=True,logical_ppo_before_eval=24,
        logical_ppo_after_eval=25,natural_ends=labels,natural_successes=wins,scorer_fit_steps=fit,
        eval_rows=outcome['rows'],eval_coverage=outcome['coverage'],budget_before=before,budget_after=after))
