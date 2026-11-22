from __future__ import annotations
import hashlib, json, os, shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def put(file,obj):
    tmp=file.with_name(file.name+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2)+'\n')
    tmp.replace(file)
def replace(text, before, after):
    if text.count(before)!=1: raise ValueError(before)
    return text.replace(before,after)
def main():
    pools=ROOT/'pools'
    pools.mkdir(exist_ok=True)
    archive=PROJECT/'archives'
    old=PROJECT/'src/fixed5d'
    manifest=json.loads((old/'pool_manifest.json').read_bytes())
    seals={}
    for name,detail in manifest['panels'].items():
        key=detail['sha256']
        dest=pools/(name+'.npz')
        if not dest.exists():
            src=archive/'objects-by-sha256'/key[:2]/key
            if not src.exists():
                for index in archive.glob('*/meta/*inventory.json'):
                    inv=json.loads(index.read_bytes())
                    entries=inv if isinstance(inv,list) else inv.get('files',inv.get('rows',[]))
                    for row in entries:
                        if isinstance(row,list) and len(row)>=3 and row[2]==key:
                            candidate=index.parents[1]/'tree'/row[0]
                            if candidate.is_file(): src=candidate; break
                    if src.exists(): break
            if not src.is_file(): raise FileNotFoundError(f'{name}: {key}')
            assert digest(src)==key
            os.link(src,dest)
        assert digest(dest)==key
        seals[name]={'sha256':key,'rows':detail['rows'],'bytes':dest.stat().st_size}
    for spec in ('sparse_schedule.json','original_index_mapping.json','pool_manifest.json'):
        shutil.copyfile(old/spec,pools/spec)
    for variant,dirname in (('neural','fixed5d'),('grid','fixed5d-grid')):
        src=PROJECT/'src'/dirname
        target=ROOT/variant;target.mkdir(exist_ok=True)
        for item in src.iterdir():
            if item.is_file() and item.suffix in ('.py','.json'): shutil.copyfile(item,target/item.name)
        fp=target/'formal_gpu4096_runtime.py'
        fp.write_text(replace(fp.read_text(),"== spec['num_envs'] == 4096","== spec['num_envs'] and env.num_envs in (2048,4096,8192,16384)"))
        fp=target/'spatial_train.py';raw=fp.read_text().split("if __name__=='__main__':")[0]
        raw=replace(raw,"        runner = Runner(", """        from tongji_runtime import bind_policy, seed_all
        seed_all(spec['seed'])
        runner = Runner(""")
        raw=replace(raw,"        if spec.get('initial_weights_sha256'):","        bind_policy(spec, out, actor, runner.alg.critic)\n        if spec.get('initial_weights_sha256'):")
        raw=replace(raw,"    arm = args.arm","    global DEVELOPMENT, FULL, modes\n    from tongji_runtime import schedule_for\n    DEVELOPMENT, FULL, modes = schedule_for(spec)\n    arm = args.arm")
        raw=replace(raw,"        for stop in range(100,total+1,100):","        for stop in sorted(set(range(100,total+1,100)) | set(DEVELOPMENT) | ({20} if spec['engineering'] else set()) | {total}):\n            prior=updates\n            batch_start=monotonic()")
        raw=replace(raw,"            runner.learn(num_learning_iterations=100,","            runner.learn(num_learning_iterations=stop-updates,")
        raw=replace(raw,"            assert updates==stop","            assert updates==stop\n            append_rows(out/'timings.jsonl',[dict(start=prior,stop=stop,training_s=monotonic()-batch_start,envs=env.num_envs)])")
        anchor="            full=stop in FULL"
        raw=replace(raw,anchor,"""            if spec['engineering'] and stop==32:
                from tongji_runtime import restore_audit
                restore_audit(checkpoint,runner,rt,ema,updates,calls,out,put)
            full=stop in FULL""")
        raw=replace(raw,"        test = evaluate(","""        if spec['engineering']:
            from tongji_runtime import bench_finish
            bench_finish(out,rt,env,runner,spec,put)
            return
        test = evaluate(""")
        raw=replace(raw,"==[7800,7900,8000]","==sorted(FULL)[-3:]")
        raw=replace(raw,"development_auc_504_sparse=interval_auc(light_curve),development_auc_interval=[500,total],","development_auc_504_sparse=sum((b['update']-a['update'])*(a['coverage']+b['coverage'])/2 for a,b in zip(light_curve,light_curve[1:]))/(total-DEVELOPMENT[0]),development_auc_interval=[DEVELOPMENT[0],total],")
        raw=replace(raw,"development_auc_normalization_updates=7500,","development_auc_normalization_updates=total-DEVELOPMENT[0],")
        raw=replace(raw,"            losses = original_update()","            opt_start=monotonic()\n            losses = original_update()\n            opt_end=monotonic()")
        raw=replace(raw,"                rt.end_update(updates)","                scorer_start=monotonic()\n                rt.end_update(updates)\n                scorer_end=monotonic()\n                if spec['engineering']: append_rows(out/'optimizer-timing.jsonl',[dict(update=updates,ppo_s=opt_end-opt_start,scorer_s=scorer_end-scorer_start)])")
        fp.write_text(raw)
    put(ROOT/'cpu-prepared.json',dict(verified_pools=seals,candidates=[2048,4096,8192,16384],formal_seeds=[72601,72602],rollout_steps=24,formal_interactions=1966080000))
if __name__=='__main__': main()

