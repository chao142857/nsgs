import numpy as np
from goalreach.curriculum import DEFAULTS,ManualCurriculum
from goalreach.p3b_sampler import LargePoolSampler
from goalreach.core import append_rows
class FixedDistribution(LargePoolSampler):
    # Exact random.choices/CDF machinery of the million-pool release with curriculum support for probability zero.
    def set_probabilities(self,values):
        p=np.asarray(values,dtype=np.float64)
        assert len(p)==self.n and np.isfinite(p).all() and (p>=0).all() and p.sum()>0
        self.external=p/p.sum()
        self._cumulative=self._probability=None
def create(method,pool,seed):
    sampler=FixedDistribution(pool['region'],seed,'external')
    curriculum=ManualCurriculum(pool['region']) if method=='manual' else None
    sampler.set_probabilities(curriculum.probabilities() if curriculum else np.ones(sampler.n))
    return sampler,curriculum
def advance(rt,update):
    curriculum=rt.curriculum
    if curriculum is None:return
    event=curriculum.end_update(update)
    if event:
        rt.sampler.set_probabilities(curriculum.probabilities())
        append_rows(rt.out/'curriculum_events.jsonl',[event])
    if update<=3 or update%50==0 or event:
        p=rt.sampler.probabilities()
        mass=np.bincount(curriculum.regions,weights=p,minlength=4)
        append_rows(rt.out/'manual_history.jsonl',[dict(update=update,stage=curriculum.stage,window=len(curriculum.history),region_mass=mass.tolist(),settings=curriculum.settings)])

