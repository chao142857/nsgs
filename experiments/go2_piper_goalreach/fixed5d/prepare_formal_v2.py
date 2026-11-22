from pathlib import Path
import shutil,json
from prepare import ROOT,PROJECT,replace,digest,put
SHAPES=(16384,32768)
VARIANTS=('neural','grid','uniform','manual')
def main():
    for slots in SHAPES:
        for variant in VARIANTS:
            parent=ROOT/('neural' if variant=='neural' else 'grid')
            d=ROOT/f'formal-v2-{variant}-{slots}';d.mkdir(exist_ok=True)
            for f in parent.iterdir():
                if f.is_file():shutil.copyfile(f,d/f.name)
            train=d/'spatial_train.py';s=train.read_text()
            if slots==32768:
                s=replace(s,"    env = make_env(cfg)","    env = make_env(cfg)\n    from packed_origin import check_env\n    check_env(env,out)")
                att=d/'formal_gpu4096_runtime.py'
                att.write_text(replace(att.read_text(),"in (2048,4096,8192,16384)","in (2048,4096,8192,16384,32768)"))
            if variant in ('uniform','manual'):
                sf=d/'spatial_runtime.py';r=sf.read_text()
                r=replace(r,"            self.sampler = sampler_for(self.method, self.pool, self.sampler_seed)","            from frozen_baselines import create\n            self.sampler,self.curriculum=create(self.method,self.pool,self.sampler_seed)")
                r=replace(r,"        if self.neural:\n            model", "        from frozen_baselines import advance\n        advance(self,update)\n        if self.neural:\n            model")
                sf.write_text(r)
                s=replace(s,"'arm_id':'neural_sharp' if arm=='nsgs_e4_sharp' else 'table_sharp'","'arm_id':arm")
                s=replace(s,"        assert np.array_equal(rt.sampler.probabilities(),np.full(len(rt.pool['task_id']),1/len(rt.pool['task_id'])))","""        actual=rt.sampler.probabilities()
        if arm=='manual':
            assert rt.curriculum.stage==0 and not np.any(actual[rt.pool['region']!=0])
            assert np.allclose(actual[rt.pool['region']==0],1/np.sum(rt.pool['region']==0),rtol=1e-14)
        else:
            assert np.array_equal(actual,np.full(len(actual),1/len(actual)))""")
                s=replace(s,"        return sha256(pickle.dumps(rt.sampler.state_dict(),protocol=5)).hexdigest()","        return sha256(pickle.dumps((rt.sampler.state_dict(),rt.curriculum.state_dict() if rt.curriculum else None),protocol=5)).hexdigest()")
                s=replace(s,"optimizer=runner.alg.optimizer.state_dict(),ema=ema,sampler=rt.sampler.state_dict(),","optimizer=runner.alg.optimizer.state_dict(),ema=ema,sampler=rt.sampler.state_dict(),curriculum=rt.curriculum.state_dict() if rt.curriculum else None,")
                att=d/'formal_gpu4096_runtime.py';a=att.read_text()
                a=replace(a,"    grid = entry(rt.pool,rt.sampler,spec['seed'])\n    put(out/'grid_entry.json',grid)", """    proof=dict(arm=args.arm,uniform_all_configs=args.arm=='uniform',manual_settings=rt.curriculum.settings if rt.curriculum else None,number_of_tasks=len(rt.pool['task_id']),seed=spec['seed'])
    put(out/'baseline_entry.json',proof)""")
                att.write_text(a)
            train.write_text(s)
    form={}
    for slots in SHAPES:
        form[str(slots)]={}
        for variant in VARIANTS:
            d=ROOT/f'formal-v2-{variant}-{slots}'
            form[str(slots)][variant]={f.name:digest(f) for f in sorted(d.glob('*.py'))}
    put(ROOT/'formal-v2-branches.json',form)
if __name__=='__main__':main()

