"""Join actual per-tick poses with collision faces and the visual USD geometry."""
from argparse import ArgumentParser
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from course import decode, mesh, tile, TERRAIN_NAMES

def prepare(evaluation, dest):
    visual=evaluation/'visuals'
    parts=json.loads((visual/'robot.json').read_text())
    bank=[]
    records=json.loads((evaluation/'strict_routes.json').read_bytes())
    if isinstance(records,dict):
        records=records.get('routes',records.get('rows'))
    by_id={int(r.get('task_id',r.get('id'))):r for r in records}
    for trace in sorted((evaluation/'traces').glob('*.npz')):
        with np.load(trace,allow_pickle=False) as v:
            task=int(v['task_id']); rec=by_id[task]
            name=trace.stem; out=dest/name;out.mkdir(exist_ok=True,parents=True)
            k,s,l,_,_=decode(task); xyz,faces=mesh(tile(k,s,l).grid)
            xyz=np.asarray(xyz,'<f4');faces=np.asarray(faces,'<u4')
            (out/'ground.mesh').write_bytes(xyz.tobytes()+faces.tobytes())
            ticks=np.array(v['t'],int); assert np.array_equal(ticks,np.arange(len(ticks)))
            poses=np.asarray(v['body_poses'],'<f4');assert poses.shape[:2]==(len(ticks),len(v['body_names']))
            assert all(str(v['body_names'][p['body']])==p['name'] for p in parts)
            assert np.isfinite(poses).all()
            # Retain first and terminal instants; encode their true timestamps even for odd terminal ticks.
            ix=np.unique(np.r_[np.arange(0,len(ticks),2),len(ticks)-1])
            (out/'poses.f32').write_bytes(poses[ix].tobytes())
            def integer(key,default=0):
                return int(rec.get(key,default))
            data=dict(folder=name,terrain=TERRAIN_NAMES[k],id=task,ticks=ticks[ix].tolist(),
                 hold=np.asarray(v['hold_count'],int)[ix].tolist(),start=v['start'].tolist(),goal=v['goal'].tolist(),
                 state=str(rec.get('result',rec.get('failure_reason','unknown'))),seed=integer('seed'),update=int(v['checkpoint']),
                 mesh=dict(file='ground.mesh',vertices=len(xyz),triangles=len(faces)),bodies=poses.shape[1])
            bank.append(data)
    assert len(bank)==len(by_id) and {r['terrain'] for r in bank}==set(TERRAIN_NAMES)
    (dest/'segments.json').write_text(json.dumps(bank,indent=2))

if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('evaluation',type=Path);p.add_argument('dest',type=Path)
    args=p.parse_args();prepare(args.evaluation,args.dest)
