"""50-byte fixed records; one NPZ per 100 completed updates, no repeated pool hash."""
import hashlib
from pathlib import Path
import numpy as np
from .core import write_json, append_rows, REASONS

DTYPE=np.dtype([('end_update','<u4'),('sample_update','<u4'),('task_id','<u4'),
    ('sample_visits','<u4'),('episode_steps','<u2'),('reason','u1'),('curriculum_stage','i1'),
    ('terminal_distance_m','<f4'),('minimum_distance_m','<f4'),
    ('sample_probability','<f8'),('sample_success_rate','<f8'),('env_id','<u2'),('generation','<u4')])

SGS_DTYPE = np.dtype(DTYPE.descr + [('sample_completed_results', '<u4'),
    ('hold_speed_lt_025', 'u1')])


class EpisodeLog:
    def __init__(self,out,pool_hash,*,sgs=False,content_hash=None,run_metadata=None):
        self.out=Path(out)/'episodes';self.out.mkdir(parents=True,exist_ok=False)
        self.rows=[];self.files=[];self.block=1
        self.record_count = 0
        self.dtype = SGS_DTYPE if sgs or run_metadata else DTYPE
        self.manifest=dict(schema='goalreach-episodes-sgs-v2' if sgs else 'goalreach-episodes-v1',
            pool_sha256=pool_hash, bytes_per_record=self.dtype.itemsize,dtype=self.dtype.descr,reason_names=REASONS,
            success='reason == 1',sample_visits='Assignments before this assignment, including incomplete episodes',
            shadow_rate='Mean of last 100 completed outcomes; unvisited prior 0.5',files=self.files)
        if content_hash:
            self.manifest['pool_content_sha256'] = content_hash
        if run_metadata and (run_metadata.get('stage')=='3-engineering' or run_metadata.get('arm_id')=='neural_sharp'):
            self.dtype=np.dtype(SGS_DTYPE.descr+[('sample_neural_rate','<f8')])
            self.manifest.update(dtype=self.dtype.descr,bytes_per_record=self.dtype.itemsize,
                                 sample_neural_rate='Ensemble probability at assignment before any mixing')
        if run_metadata:
            self.manifest.update(run_metadata)
            schemas={'3-engineering':'goalreach-neural-engineering','P3B1':'goalreach-episodes-P3B1-v1',
                     'P3S1':'goalreach-episodes-P3S1-v1', 'P3S2':'goalreach-episodes-P3S2-v1'}
            self.manifest['schema'] = schemas.get(run_metadata.get('stage'),'goalreach-episodes-K2')
        if run_metadata and run_metadata.get('arm_id') in ('grid16_sharp','grid32_sharp'):
            self.manifest['shadow_rate'] = 'Mean of last 100 natural endings in the assigned geometry cell; empty prior 0.5'
        if sgs or run_metadata:
            self.manifest.update(sample_completed_results='Natural training endings before assignment',
                hold_speed_lt_025='Success with maximum hold speed strictly below 0.25 m/s')
        self.save_manifest()

    def save_manifest(self):
        temporary = self.out/'manifest.tmp'
        write_json(temporary, self.manifest)
        temporary.replace(self.out/'manifest.json')
    def add(self,row):
        reason=1 if row['success'] else REASONS.index(row['failure_reason'])
        self.rows.append(tuple(row[k] if k!='reason' else reason for k in self.dtype.names))
        self.record_count += 1
    def flush(self,update,final=False):
        if not final and update%100:return
        if not self.rows:return
        array=np.array(self.rows,dtype=self.dtype)
        path=self.out/f'{self.block:04d}_{update:06d}.npz'
        np.savez_compressed(path,episodes=array)
        self.files.append(dict(path=path.name,records=len(array),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),through_update=update))
        self.rows=[];self.block+=1;self.save_manifest()

class SamplingDiagnostics:
    def __init__(self, *, sgs=False):
        self.sgs = sgs
        self.boundary_assignments = 0
        self.regions=np.zeros(4,np.int64);self.rates=np.zeros(6,np.int64)
    def add(self,region,rate,completed):
        self.regions[region]+=1
        group=5 if completed==0 else 0 if rate==0 else 4 if rate==1 else 1 if rate<.3 else 2 if rate<.7 else 3
        self.rates[group]+=1
        if self.sgs and completed and .3 <= rate <= .7:
            self.boundary_assignments += 1
    def export(self,sampler,update):
        p=np.asarray(sampler.probabilities());q=p[p>0]
        result = dict(update=update,entropy=float(-(q*np.log(q)).sum()),
            effective_configurations=float(1/(p*p).sum()),effective_definition='1 / sum(p**2)',
            regional_assignments=self.regions.tolist(),
            regional_shares=(self.regions/max(1,self.regions.sum())).tolist(),
            shadow_rate_groups=['0','(0,.3)','[.3,.7)','[.7,1)','1','unvisited'],
            shadow_rate_counts=self.rates.tolist(),
            visits_below_three=int((np.asarray(sampler.sample_counts)<3).sum()))

        if self.sgs:
            completed = np.asarray(sampler.completed_counts)
            sampled = np.asarray(sampler.sample_counts)
            boundary = (completed > 0) & (sampler.cached_rates >= .3) & (sampler.cached_rates <= .7)
            labeled_assignments = int(self.regions.sum() - self.rates[-1])
            result.update(regional_probability_mass=np.bincount(sampler.regions, weights=p, minlength=4).tolist(),
                never_sampled_fraction=float((sampled == 0).mean()),
                no_completed_result_fraction=float((completed == 0).mean()),
                configurations_with_fewer_than_three_results=int((completed < 3).sum()),
                boundary_fraction_of_sampled_configurations=float(boundary.sum()/max(1, (sampled > 0).sum())),
                boundary_fraction_of_labeled_configurations=float(boundary.sum()/max(1, (completed > 0).sum())),
                boundary_fraction_of_labeled_assignments=self.boundary_assignments/labeled_assignments if labeled_assignments else None,
                boundary_probability_mass=float(p[boundary].sum()),
                completed_results=int(completed.sum()), assignments=int(sampled.sum()))
        return result
