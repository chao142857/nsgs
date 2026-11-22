import math
from pathlib import Path
import unittest
import numpy as np
import torch
from goalreach.core import EpisodeState, Sampler, inverse_rotate, body_vectors, failure_reward
from goalreach.vendor_sampler import TaskSampler
from goalreach.kinematics import Arm
from tools.build_pools import make_pool, content_hash, COUNTS


class Semantics(unittest.TestCase):
    def test_timeout_penalty_switch_preserves_physical_failures(self):
        from types import SimpleNamespace
        from goalreach.mdp import terminal_failure
        reasons=torch.arange(6)
        torch.testing.assert_close(failure_reward(reasons),torch.tensor([0.,0.,-100.,-100.,-100.,-100.]))
        torch.testing.assert_close(failure_reward(reasons,-50),torch.tensor([0.,0.,-50.,-100.,-100.,-100.]))
        with self.assertRaises(ValueError):failure_reward(reasons,-25)
        env=SimpleNamespace(step_dt=.02,goal_runtime=SimpleNamespace(state=SimpleNamespace(reason=reasons)),
            cfg=SimpleNamespace(goalreach={'timeout_penalty':-50.}))
        torch.testing.assert_close(terminal_failure(env)*(-100)*env.step_dt,
            torch.tensor([0.,0.,-50.,-100.,-100.,-100.]))
    def setUp(self):
        self.s=EpisodeState(1)
        self.s.reset([0],torch.tensor([7]),torch.tensor([[.1,.2,.3]]))
        self.z=torch.zeros(1,3);self.no=torch.zeros(1,dtype=torch.bool)
    def step(self,ee=None,vel=None,base=False,tip=False,nonfoot=False):
        return self.s.advance(self.s.target if ee is None else ee,self.z if vel is None else vel,
            torch.tensor([base]),torch.tensor([tip]),torch.tensor([nonfoot]))
    def test_24_25_once_and_feedback(self):
        for _ in range(24):self.assertFalse(self.step().item())
        self.assertTrue(self.step().item())
        row=self.s.rows([0])[0]
        self.assertTrue(row['success']);self.assertEqual(row['episode_steps'],25)
        self.assertEqual(row['task_id'],7);self.assertEqual(row['minimum_distance_m'],0)
        self.step();self.assertEqual(self.s.steps.item(),25)
    def test_pass_through_speed_and_streak_restart(self):
        for _ in range(24):self.step()
        self.step(vel=torch.tensor([[.25,0.,0.]]));self.assertEqual(self.s.hold.item(),0)
        for _ in range(24):self.assertFalse(self.step().item())
        self.assertTrue(self.step().item())
    def test_hold_only_success_and_speed_record(self):
        for criterion in ('hold_only', 'hold_and_speed'):
            with self.subTest(criterion=criterion):
                self.s=EpisodeState(1,success_criterion=criterion)
                self.s.reset([0],torch.tensor([7]),torch.zeros(1,3))
                for _ in range(24):
                    self.assertFalse(self.step(vel=torch.tensor([[.4,0.,0.]])).item())
                self.assertEqual(self.step(vel=torch.tensor([[.9,0.,0.]])).item(),criterion=='hold_only')
                if criterion=='hold_only':
                    self.step()
                    row=self.s.rows([0])[0]
                    self.assertEqual(row['episode_steps'],25)
                    self.assertAlmostEqual(row['hold_speed_max_m_s'],.9,6)
                    self.assertAlmostEqual(row['hold_speed_mean_m_s'],.42,6)
    def test_high_speed_crossing_fails_both_criteria(self):
        for criterion in ('hold_only', 'hold_and_speed'):
            with self.subTest(criterion=criterion):
                self.s=EpisodeState(1,success_criterion=criterion)
                self.s.reset([0],torch.tensor([7]),torch.zeros(1,3))
                for x in (-.08,-.04,0.,.04,.08):
                    self.assertFalse(self.step(ee=torch.tensor([[x,0.,0.]]),
                                               vel=torch.tensor([[2.,0.,0.]])).item())
                self.assertEqual(self.s.hold.item(),0)
    def test_timeout_no_bootstrap_semantics(self):
        for _ in range(499):self.assertFalse(self.step(ee=self.z+10).item())
        self.assertTrue(self.step(ee=self.z+10).item())
        self.assertEqual(self.s.rows([0])[0]['failure_reason'],'timeout')
    def test_failures_override_success(self):
        for _ in range(24):self.step()
        self.step(base=True,tip=True);self.assertEqual(self.s.reason.item(),3)
        self.setUp();self.step(tip=True);self.assertEqual(self.s.reason.item(),4)
        self.setUp();self.step(nonfoot=True);self.assertEqual(self.s.reason.item(),5)
    def test_fixed_target_and_reset(self):
        old=self.s.target.clone()
        for _ in range(100):self.step(torch.randn(1,3),torch.randn(1,3))
        self.assertTrue(torch.equal(old,self.s.target))
        self.s.reset([0],torch.tensor([8]),old+1)
        self.assertTrue(torch.equal(old+1,self.s.target));self.assertEqual(self.s.hold.item(),0)
    def test_yaw_pitch_hand_values(self):
        # Rz(90) Ry(30). Independent hand-expanded rotation matrix.
        a=math.sqrt(3)/2;b=.5
        R=torch.tensor([[0.,-1.,0.],[a,0.,b],[-b,0.,a]],dtype=torch.float64)
        q=torch.tensor([[math.cos(math.pi/4)*math.cos(math.pi/12),
             -math.sin(math.pi/4)*math.sin(math.pi/12),
             math.cos(math.pi/4)*math.sin(math.pi/12),
             math.sin(math.pi/4)*math.cos(math.pi/12)]],dtype=torch.float64)
        base=torch.tensor([[1.,2.,.35]],dtype=torch.float64)
        ee=base+torch.tensor([[.1,.2,.1]],dtype=torch.float64)
        target=base+torch.tensor([[.3,-.4,.2]],dtype=torch.float64)
        v1,v2=body_vectors(q,base,ee,target)
        torch.testing.assert_close(v1,(target-base)@R)
        torch.testing.assert_close(v2,(target-ee)@R)
        torch.testing.assert_close(inverse_rotate(q,torch.tensor([[0.,1.,0.]],dtype=torch.float64)),torch.tensor([[a,0.,b]],dtype=torch.float64))
        far,_=body_vectors(q,base,ee,base+torch.tensor([[0.,3.,0.]],dtype=torch.float64))
        torch.testing.assert_close(far,torch.tensor([[a,0.,b]],dtype=torch.float64))
    def test_progress_cap(self):
        self.step(ee=self.s.target+1)
        self.step(ee=self.s.target)
        self.assertAlmostEqual(self.s.progress.item(),.02,6)


class Sampling(unittest.TestCase):
    def test_sgs_equivalence_and_history(self):
        a=Sampler([0,1,2,3],5,'sgs');b=TaskSampler(4,5,'sgs')
        for i in range(111):
            ids=[0,1,2,3];wins=[True,i%2==0,False,True]
            a.record(ids,wins);b.record(ids,wins)
        np.testing.assert_allclose(a.probabilities(),b.probabilities(),rtol=1e-14)
        self.assertEqual(a.sample(100),b.sample(100));self.assertEqual(len(a.outcomes[0]),100)
        c=Sampler([0,1,2,3],9,'sgs');c.load_state_dict(a.state_dict())
        self.assertEqual(a.sample(30),c.sample(30))
    def test_external_and_region(self):
        s=Sampler([0,0,1,2],3,'external')
        for p in ([1,2],[0,0,0,0],[1,-1,1,1],[1,1,float('nan'),1]):
            with self.assertRaises(ValueError):s.set_probabilities(p)
        s.set_probabilities([0,1,0,0]);self.assertEqual(s.sample(100),[1]*100)
        s=Sampler([0,0,1,2],3,'uniform','near');self.assertEqual(set(s.sample(100)),{0,1})
    def test_evaluation_no_feedback_update(self):
        s=Sampler([0,1,2,3],3,'evaluation');self.assertEqual(s.sample(6),[0,1,2,3,0,1])
        s.record([0],[True]);self.assertEqual(s.completed_counts,[0]*4)


class Pools(unittest.TestCase):
    def test_pool_contract_and_witness(self):
        arm=Arm(Path(__file__).resolve().parents[1]/'reference/native_kinematics.json')
        a=make_pool(arm,2026111207);b=make_pool(arm,2026111208,True)
        self.assertEqual(content_hash(a),content_hash(make_pool(arm,2026111207)))
        self.assertEqual(len(a['target']),32768);self.assertEqual(len(b['target']),1008)
        self.assertEqual(np.bincount(a['region']).tolist(),COUNTS)
        for r in range(4):
            for z in range(3):self.assertEqual(int(((b['region']==r)&(b['height_stratum']==z)).sum()),84)
        self.assertFalse(set(map(tuple,a['target']))&set(map(tuple,b['target'])))
        for d in [a,b]:
            self.assertTrue((d['initial_joint_delta'][:,13]>=0).all())
            self.assertTrue((d['initial_joint_delta'][:,14]<=0).all())
            self.assertEqual(len(d['target']),len(np.unique(d['target'],axis=0)))
            self.assertTrue((d['horizontal_distance']<=3).all())
            self.assertTrue(((d['height']>=.1)&(d['height']<=.85)).all())
            self.assertTrue((d['nominal_distance'][d['region']==0]<=.050001).all())
            q=d['inplace_joint_witness'][d['region']==1]
            np.testing.assert_allclose(arm.fk(q),d['target'][d['region']==1],atol=3e-7)
            self.assertTrue(arm.admissible(q).all())


if __name__=='__main__':unittest.main()
