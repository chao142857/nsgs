"""50-byte fixed records; one NPZ per 100 completed updates, no repeated pool hash."""
import hashlib
from pathlib import Path
import numpy as np
from .core import write_json, append_rows, REASONS

DTYPE=np.dtype([('end_update','<u4'),('sample_update','<u4'),('task_id','<u4'),
    ('sample_visits','<u4'),('episode_steps','<u2'),('reason','u1'),('curriculum_stage','i1'),
    ('terminal_distance_m','<f4'),('minimum_distance_m','<f4'),
    ('sample_probability','<f8'),('sample_success_rate','<f8'),('env_id','<u2'),('generation','<u4')])

class EpisodeLog:
    def __init__(self,out,pool_hash):
        self.out=Path(out)/'episodes';self.out.mkdir(parents=True,exist_ok=False)
        self.rows=[];self.files=[];self.block=1
        self.manifest=dict(schema='goalreach-episodes-v1',pool_sha256=pool_hash,
            bytes_per_record=DTYPE.itemsize,dtype=DTYPE.descr,reason_names=REASONS,
            success='reason == 1',sample_visits='Assignments before this assignment, including incomplete episodes',
            shadow_rate='Mean of last 100 completed outcomes; unvisited prior 0.5',files=self.files)
        self.save_manifest()
    def save_manifest(self):write_json(self.out/'manifest.json',self.manifest)
    def add(self,row):
        reason=1 if row['success'] else REASONS.index(row['failure_reason'])
        self.rows.append(tuple(row[k] if k!='reason' else reason for k in DTYPE.names))
    def flush(self,update,final=False):
        if not final and update%100:return
        if not self.rows:return
        array=np.array(self.rows,dtype=DTYPE)
        path=self.out/f'{self.block:04d}_{update:06d}.npz'
        np.savez_compressed(path,episodes=array)
        self.files.append(dict(path=path.name,records=len(array),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),through_update=update))
        self.rows=[];self.block+=1;self.save_manifest()

class SamplingDiagnostics:
    def __init__(self):
        self.regions=np.zeros(4,np.int64);self.rates=np.zeros(6,np.int64)
    def add(self,region,rate,completed):
        self.regions[region]+=1
        group=5 if completed==0 else 0 if rate==0 else 4 if rate==1 else 1 if rate<.3 else 2 if rate<.7 else 3
        self.rates[group]+=1
    def export(self,sampler,update):
        p=np.asarray(sampler.probabilities());q=p[p>0]
        return dict(update=update,entropy=float(-(q*np.log(q)).sum()),
            effective_configurations=float(1/(p*p).sum()),effective_definition='1 / sum(p**2)',
            regional_assignments=self.regions.tolist(),
            regional_shares=(self.regions/max(1,self.regions.sum())).tolist(),
            shadow_rate_groups=['0','(0,.3)','[.3,.7)','[.7,1)','1','unvisited'],
            shadow_rate_counts=self.rates.tolist(),
            visits_below_three=int((np.asarray(sampler.sample_counts)<3).sum()))
