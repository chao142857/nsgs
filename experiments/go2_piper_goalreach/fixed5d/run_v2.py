import argparse, json, os, sys, traceback
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
p=argparse.ArgumentParser()
p.add_argument('mode',choices=['benchmark','formal'])
p.add_argument('variant',choices=['neural','grid','uniform','manual'])
p.add_argument('seed',type=int)
p.add_argument('slots',type=int,choices=(2048,4096,8192,16384,32768))
p.add_argument('out',type=Path)
x=p.parse_args()
def main():
    assert x.mode=='formal' and x.slots in (16384,32768)
    branch=ROOT/f'formal-v2-{x.variant}-{x.slots}'
    sys.path.insert(0,str(branch))
    from spatial_train import run,put
    from protocol import file_sha,validate
    from tongji_runtime import schedule_for
    phys=json.loads((PROJECT/'receipts/physics.json').read_text())
    assert phys['passed'] and (PROJECT/'receipts/deploy.exit').read_text().strip()=='0'
    from approve_v2 import permit
    go=permit()
    assert x.seed in go['fixed5d_seeds'] and x.slots==go['fixed5d_parallel_envs']
    spec=json.loads((branch/'spec_seed43.json').read_bytes())
    spec.pop('initial_weights_file',None);spec.pop('initial_weights_sha256',None);spec.pop('actor_sha256',None)
    arm={'neural':'nsgs_e4_sharp','grid':'grid_selected_8x8x4x16x8_sharp','uniform':'uniform','manual':'manual'}[x.variant]
    spec.update(seed=x.seed,num_envs=x.slots,updates_per_arm=33 if x.mode=='benchmark' else 81920000//x.slots,engineering=x.mode=='benchmark',selected_arm=arm)
    assert spec['engineering'] or spec['updates_per_arm']*24*x.slots==1966080000
    pools=ROOT/'pools'
    panel, mapping, oldsha=validate(pools)
    for name,key in spec['pool_sha256'].items(): assert file_sha(pools/(name+'.npz'))==key
    nodes,full,modes=schedule_for(spec)
    schedule=dict(original=oldsha,nodes=nodes,methods={str(u):modes(u) for u in nodes},full=sorted(full),formal_interactions=1966080000,variant=x.variant,seed=x.seed,slots=x.slots)
    put(x.out/'schedule.json',schedule)
    args=SimpleNamespace(arm=arm,out=x.out,pools=pools,compact_panel=panel,original_mapping=mapping,schedule_sha=file_sha(x.out/'schedule.json'),audit_entry=False,learning_probe=False,grid_development=False,formal_started=monotonic())
    from isaaclab.app import AppLauncher
    kits=['--/app/extensions/registryEnabled=false','--/renderer/multiGpu/enabled=false','--/renderer/multiGpu/autoEnable=false','--/app/asyncRendering=false','--/app/file/ignoreUnsavedOnExit=true','--/plugins/carb.tasking.plugin/threadCount=2','--/plugins/omni.tbb.globalcontrol/maxThreadCount=2','--/plugins/carb.scripting-python.plugin/disableDelayloads=true',f'--/log/file={x.out}/kit.log']
    kits.extend(f'--/persistent/isaac/asset_root/{k}={PROJECT}/src/static/assets' for k in ('local','cloud','nvidia','default'))
    kits.append('--/renderer/activeGpu='+os.environ.get('KIT_PHYSICAL_GPU','0'))
    sys.argv[:]=sys.argv[:1]
    app=AppLauncher(headless=True,device='cuda:0',enable_cameras=False,livestream=0,experience=str(PROJECT/'src/static/lab/apps/isaaclab.python.headless.kit'),kit_args=' '.join(kits)).app
    try:
        if x.slots==32768:
            from packed_origin import install
            install(x.out)
        run(args,spec)
        from physics_log_audit import audit
        proof=audit(x.out)
        if not proof['passed']:
            result=x.out/'result.json'
            if result.is_file(): result.rename(x.out/'pre-log-screen-result.json')
            raise RuntimeError(proof)
    except BaseException:
        (x.out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        app.close(wait_for_replicator=False,skip_cleanup=True)
if __name__=='__main__':
    try: main()
    except BaseException:
        (x.out/'error.txt').write_text(traceback.format_exc())
        raise

