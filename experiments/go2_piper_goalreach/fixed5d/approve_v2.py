import json
from pathlib import Path
from prepare import digest
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
def permit():
    go=json.loads((PROJECT/'experiments/release-20261122/formal-go-v2.json').read_text())
    scale=json.loads((ROOT/'formal-v2-scale.json').read_text())
    assert go['approved'] is True and go['schema']==2 and go['fixed5d_seeds']==[72601,72602]
    slots=go['fixed5d_parallel_envs']
    assert slots in (16384,32768) and go['fixed5d_updates_per_arm']==81920000//slots
    assert go['interactions_per_arm']==1966080000==slots*go['fixed5d_updates_per_arm']*24
    assert scale['completed'] and scale['selected_envs']==slots
    assert scale['comparison_sha256']==digest(ROOT/'benchmark-32768-comparison.json')
    assert scale['passed_32768'] is (slots==32768)
    return go

