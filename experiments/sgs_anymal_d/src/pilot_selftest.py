"""CPU-only checks for the frozen training threshold, table and update-accounting contract."""
from collections import deque
from pathlib import Path
import tempfile
import unittest
import numpy as np

import navigation
from pilot_config import atomic_json, check_physics, locked_budget, reserve
from table import Table
from train import record_groups


class PilotTests(unittest.TestCase):
    def test_training_arrival_and_forward_clock(self):
        navigation.controlled_cases()
        self.assertIsNone(navigation.terminal(.60,0.,19,30))
        self.assertEqual(navigation.terminal(.60,0.,20,30),'success')
        self.assertEqual(navigation.terminal(.60,42.,20,30),'base_contact')
        self.assertEqual(navigation.reward_weights(499)[-1],.3)
        self.assertEqual(navigation.reward_weights(500)[-1],0.)

    def test_same_tick_duplicate_ids(self):
        rng=np.random.default_rng(187)
        table=Table(8)
        rows=[deque(maxlen=100) for _ in range(8)]
        self.assertTrue(np.all(table.mass==table.weights(np.array([.5]))[0]))
        for _ in range(20):
            ids=rng.integers(0,7,512)
            ids[:180]=0
            results=rng.integers(0,2,len(ids),dtype=np.uint8)
            for idx,success in zip(ids,results):
                rows[idx].append(int(success))
            table.record(ids,results)
            p=np.array([sum(row)/len(row) if row else .5 for row in rows])
            mass=np.sqrt(np.maximum((p+1.e-4)**3.3*(1.-p+1.e-4)**1.7,1.e-4)+1.e-4)
            np.testing.assert_array_equal(table.total,[sum(row) for row in rows])
            np.testing.assert_allclose(table.mass,mass,rtol=1.e-12)
            self.assertTrue((table.mass>0).all())

    def test_cumulative_budget_and_bank_guard(self):
        with tempfile.TemporaryDirectory() as work:
            work=Path(work)
            run=dict(manifest='pilot-1',git_commit='source',geometry_hashes=['train','heldout'])
            atomic_json(work/'B.json',dict(passed=True,zero_ppo=True,git_commit='source',geometry_hashes=['train','heldout']))
            check_physics(work/'B.json',run)
            with self.assertRaises(ValueError):
                check_physics(work/'B.json',dict(run,geometry_hashes=['old','heldout']))
            with locked_budget(work/'ledger',run,work/'first',False) as (path,budget):
                budget['reserved']=499
                self.assertTrue(reserve(path,budget))
                self.assertFalse(reserve(path,budget))
            with self.assertRaises(RuntimeError),locked_budget(work/'ledger',run,work/'first',False):
                pass
            with locked_budget(work/'ledger',run,work/'first',True) as (path,budget):
                self.assertFalse(reserve(path,budget))

    def test_counts_respect_terrain_class_and_outcome(self):
        terrains=np.repeat([0,1],3)
        classes=np.tile(np.arange(3),2)
        ids=np.array([1,1,1,5])
        outcomes=np.array([1,2,4,4],dtype=np.int8)
        draws=record_groups(terrains,classes,ids)
        counts=record_groups(terrains,classes,ids,outcomes)
        self.assertEqual(draws[0,1],3)
        self.assertEqual(counts[0,1].tolist(),[0,1,1,0,1])
        self.assertEqual(counts[1,2,4],1)
        np.testing.assert_array_equal(counts.sum(axis=2),draws)


if __name__=='__main__':
    unittest.main()
