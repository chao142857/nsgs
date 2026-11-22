"""Build the isolated simulation runtime beside the prior sealed experiment."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

def replace(s,old,new):
    if s.count(old)!=1:raise ValueError('Unexpected shared source: '+old[:90])
    return s.replace(old,new)


def prepare(root,old):
    src=root/'src'
    for name in ('ppo_smoke.py','pool.py','navigation.py'):
        shutil.copyfile(old/name,src/name)
    (src/'catalog.py').write_text("from pathlib import Path\nROOT=Path(__file__).resolve().parent.parent\n")
    (src/'terrain.py').write_text("from course import *\n")
    # Preserve the native CPU ring-buffer handling for repeated IDs within one frame.
    old_sampler=(old/'sampling.py').read_text()
    table=old_sampler[old_sampler.index('class Table:'):old_sampler.index('class Distribution:')]
    table=replace(table,'e=1e-8','e=1e-4')
    (src/'table.py').write_text('import numpy as np\nfrom course import TASK_COUNT\n\n'+table)
    nav=(src/'navigation.py').read_text().replace('SUCCESS_HOLD = 5\n','SUCCESS_HOLD = 20\n')
    (src/'navigation.py').write_text(nav)
    s=(old/'scene.py').read_text()
    s=replace(s,'        self.hold = torch.where(near, self.hold + 1, 0)\n','')
    s=replace(s,'        touchdown = foot_contact & ~self.previous_contact', '''        from diagnostics import stable_arrival
        stable = stable_arrival(near, state.projected_gravity_b[:,2],
                                state.root_lin_vel_w[:,:2].norm(dim=-1),
                                state.root_ang_vel_w.norm(dim=-1), foot_contact.sum(dim=1))
        self.hold = torch.where(stable,self.hold+1,0)
        touchdown = foot_contact & ~self.previous_contact''')
    (src/'scene.py').write_text(s)
    s=(old/'native.py').read_text()
    s=replace(s,'        self.robot.write_root_pose_to_sim(roots[:,:7],env_ids=slots)', '''        # A start pose is independent of the goal. Face the obstacle from either apron.
        yaw = torch.where(g[:,0]<0,0.,torch.pi) + .20*torch.sin(13*g[:,0]+7*g[:,1])
        roots[:,3:7]=torch.stack((torch.cos(yaw/2),torch.zeros_like(yaw),torch.zeros_like(yaw),torch.sin(yaw/2)),dim=1)
        self.robot.write_root_pose_to_sim(roots[:,:7],env_ids=slots)''')
    s=replace(s,'        self.last_action[slots]=0.', '''        self.robot.set_joint_position_target(self.robot.data.default_joint_pos[slots],env_ids=slots)
        self.last_action[slots]=0.''')
    (src/'native.py').write_text(s)
    for f in src.glob('*.py'):compile(f.read_text(),str(f),'exec')
    from course import build
    from table import Table
    import numpy as np
    a=Table(6);rng=np.random.default_rng(9701);hist=[[] for _ in range(6)]
    for _ in range(12):
        ids=rng.integers(0,6,480);ids[:200]=0;ys=rng.integers(0,2,len(ids),dtype=np.uint8)
        a.record(ids,ys)
        for i,y in zip(ids,ys):hist[i].append(int(y))
    ps=np.array([np.mean(rows[-100:]) for rows in hist]);eps=1.e-4
    expected=np.sqrt(np.maximum((ps+eps)**3.3*(1-ps+eps)**1.7,eps)+eps)
    np.testing.assert_allclose(a.mass,expected,rtol=1.e-12)
    for i,rows in enumerate(hist):assert a.seen[i]==len(rows) and a.total[i]==sum(rows[-100:])
    cat=build(root)
    assert cat['distinct_raster_hashes']==260
    sources={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in src.glob('*.py')}
    protocol=dict(edition='SGS independent locomotion replication from a clean policy', robot='ANYmal-D',
        envs=32768,updates=5000,seeds=[73201,73202,73203,73204],steps_per_update=24,
        controls_per_seed=3932160000,paper='https://arxiv.org/html/2610.12465',
        author_demos='https://sgs-rl.github.io/',sampler=dict(target=.66,kappa=5,history=100,epsilon=eps,temperature=2,empty_rate=.5),
        terrain=cat,actors_from_scratch=True,maximum_host_threads_per_worker=4,
        local_choices=['procedural terrain meshes and sampled start/goal patches; exact author generator not public',
                       '20 starts and 20 goals per tile, half crossing; 20 hold ticks with upright, low speed and two-foot support',
                       'same family/level/terrain seeds, independent continuous held-out start and goal coordinates, 2080 tasks',
                       'public ANYmal-D asset and public IsaacLab learned actuator originally identified on ANYmal-C',
                       'PPO hyperparameters and physics step inherited from tested Isaac/RSL-RL stack; forward coefficient removed at update 500'],
        quality_outputs=['preselected cross-terrain body poses for each of 13 families','all-panel strict stable arrivals','cross-only success, falls and timeouts'],
        literature_reference='32K SGS final 0.494 +/- 0.113, three seeds; evaluation and unknown thresholds prevent equating success numbers.',
        code_sha256=sources,expected_u0_labels=0)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    return protocol


if __name__=='__main__':
    root=Path(__file__).resolve().parent.parent
    p=prepare(root,Path(sys.argv[1]))
    print(json.dumps({k:p[k] for k in ('envs','updates','seeds','sampler','terrain')}))
