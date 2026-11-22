"""Compare sparse anisotropic grids with identical feedback and interaction budgets."""
import json

CANDIDATES = ((8,8,8,8,8),(16,8,8,4,8),(8,8,4,8,16),(8,8,4,16,8))
SENSITIVITY = (16,16,16,4,4)
DEVELOPMENT_GRIDS = {'grid_'+'x'.join(map(str,node)):node for node in (*CANDIDATES,SENSITIVITY)}
FEEDBACK_BUDGET = 12000
EVALUATION_UPDATES = (0,100,250,500)


def development_sampler(pool, seed, bins):
    import numpy as np
    from samplers import SparseGridSampler

    class DevelopmentGrid(SparseGridSampler):
        def __init__(self):
            super().__init__(pool,seed,5,bins)
            self.feedback_labels=0
            self.natural_ends=0
            self.natural_positives=0
            self.labelled_positives=0
            self.full_labels=np.zeros(len(self.occupancy),dtype=np.uint32)
            self.full_positives=np.zeros_like(self.full_labels)

        def record(self, ids, successes, end_updates):
            count=len(ids)
            accepting=min(count,FEEDBACK_BUDGET-self.feedback_labels)
            chosen=np.asarray(ids,dtype=np.intp)
            np.add.at(self.full_labels,self.ids[chosen],1)
            np.add.at(self.full_positives,self.ids[chosen],np.asarray(successes,dtype=np.uint32))
            self.natural_ends+=count
            self.natural_positives+=sum(successes)
            super().record(ids[:accepting],successes[:accepting],end_updates[:accepting])
            self.feedback_labels+=accepting
            self.labelled_positives+=sum(successes[:accepting])

        def grid_cost(self):
            visits=np.zeros(len(self.occupancy),dtype=np.uint64)
            np.add.at(visits,self.ids,self.sample_counts)
            metrics={'candidate':list(bins),'total_bins':int(np.prod(bins)),
                     'occupied_bins':len(self.occupancy),'cell_sha256':self.cell_ids_sha256,
                     'visited_configurations':int(np.count_nonzero(self.sample_counts)),
                     'visits_total':int(visits.sum()),'visited_cells':int(np.count_nonzero(visits)),
                     'natural_ending_labels':self.natural_ends,'natural_positive_labels':self.natural_positives,
                     'natural_ending_cells':int(np.count_nonzero(self.full_labels)),
                     'positive_cells':int(np.count_nonzero(self.full_positives)),
                     'estimator_label_cap':FEEDBACK_BUDGET,'estimator_labels':self.feedback_labels,
                     'estimator_positive_labels':self.labelled_positives}
            return {key:int(value) if isinstance(value,np.integer) else value for key,value in metrics.items()}

    return DevelopmentGrid()


def run_grid_development(args,out,rt,env,wrapped,runner,ema,spec):
    from spatial_train import evaluate, put

    registered=json.loads((args.evaluation_package.parent/'development_preregistration.json').read_bytes())['grid_selection']
    assert [list(x) for x in CANDIDATES]==registered['candidates']
    assert registered['feedback_label_budget_per_candidate']==FEEDBACK_BUDGET
    assert registered['evaluation_ppo']==list(EVALUATION_UPDATES)
    budget=out/'ppo_budget.json'
    prior=0
    curve=[]
    for update in EVALUATION_UPDATES:
        if update>prior:
            runner.current_learning_iteration=prior
            runner.learn(num_learning_iterations=update-prior,init_at_random_ep_len=False)
            rt.episode_log.flush(update,final=True)
        before=budget.read_bytes()
        current=json.loads(before)
        assert current['ppo_updates_completed']==update and current['optimizer_calls']==20*update
        counts=rt.sampler.grid_cost()
        score=evaluate(env,wrapped,runner,ema,args.compact_panel,update,out,rt,spec,fixed_ema_only=True)
        after=rt.sampler.grid_cost()
        visits=('visited_configurations','visits_total','visited_cells')
        assert score['rows']==504 and budget.read_bytes()==before
        assert all(after[key]==value for key,value in counts.items() if key not in visits)
        births=after['visits_total']-counts['visits_total']
        assert births==env.num_envs,births
        curve.append(dict(score,feedback=counts,training_births_after_isolated_eval=births))
        put(out/'development_curve.json',curve)
        prior=update
    cost=rt.sampler.grid_cost()
    assert cost['natural_ending_labels']==rt.episode_log.record_count
    exact_label_budget=cost['estimator_labels']==FEEDBACK_BUDGET
    value=sum((b['update']-a['update'])*(a['coverage']+b['coverage'])/2 for a,b in zip(curve,curve[1:]))/500
    put(out/'passed.json',dict(passed=True,eligible_for_selection=exact_label_budget,development_only=True,
        development_seed=spec['seed'],candidate=list(DEVELOPMENT_GRIDS[args.arm]),fixed_coverage_auc=value,
        ppo_updates=500,interactions=500*24*256,grid_cost=cost,curve=curve))
