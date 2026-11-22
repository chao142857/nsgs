"""Combine SGS replication results at common completed deterministic evaluations."""
from pathlib import Path
from statistics import mean, stdev
import json
import math
import os
import sys
import time


def read(p): return json.loads(p.read_bytes())


def update(root):
    protocol=read(root/'protocol.json')
    lanes=[]; running=0; errors=[]
    for card,seed in enumerate(protocol['seeds']):
        lane_file=root/f'lane{card}.json'
        if not lane_file.is_file():
            return False
        lane=read(lane_file); assert lane['seed']==seed
        p=Path(lane['out']); end=p/'exit.json'
        if not end.is_file():
            running+=1
        elif not read(end)['passed']:
            errors.append(card)
        snaps={int(f.parent.name[1:]):read(f) for f in (p/'evaluations').glob('u*/summary.json')}
        lanes.append(snaps)
    common=sorted(set.intersection(*(set(s) for s in lanes)))
    result=dict(dataset='fresh reset coordinates on the frozen layouts',seeds=protocol['seeds'],
                envs=protocol['envs'],running=running,failed_lanes=errors,common={})
    for node in common:
        rows=[s[node] for s in lanes]
        rates=[s['success']/s['n'] for s in rows]
        cross=[s['cross_stable_success']/s['cross_n'] for s in rows]
        half=3.182446305*stdev(rates)/math.sqrt(4)
        result['common'][node]=dict(seeds=[dict(seed=s['seed'],success=s['success'],n=s['n'],
            cross_success=s['cross_stable_success'],cross_n=s['cross_n']) for s in rows],
            mean_success=mean(rates),seed_t95_half_width=half,mean_crossing_success=mean(cross),
            paper_32k=0.494,paper_32k_ci95_half_width=0.113)
    result['complete_four_seeds']=not running and not errors and 5000 in common
    result['observed_unix']=time.time()
    tmp=root/'combined.pending';tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(root/'combined.json')
    return not running


if __name__=='__main__':
    p=Path(sys.argv[1]).resolve()
    while not update(p):
        if '--once' in sys.argv:break
        time.sleep(50)
