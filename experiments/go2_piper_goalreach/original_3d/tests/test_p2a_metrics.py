import unittest
import numpy as np
from goalreach.p2a_metrics import arrival,forgetting,boundary,physical_and_coordination,curve,summarize,compare
from tools.build_p2a_pools import cells

class MetricsTests(unittest.TestCase):
    def test_arrival_and_never(self):
        self.assertEqual(arrival([100,200,300,400],[.1,.5,.6,.5]),200)
        self.assertIsNone(arrival([100,200,300],[.5,.49,.7]))
    def test_forgetting(self):
        a=[[.79,.5,.8],[.8,.4,.9],[1.,.3,.7],[.6,.2,.8]]
        d = forgetting(a)
        self.assertAlmostEqual(d['mean'], .3)
        self.assertEqual(d['cells_drop_over_20pp'], 1)
        self.assertEqual(d['cells_reached_80'], 2)
        self.assertEqual(d['never_reached_80'], 1)
        self.assertIsNone(d['per_cell'][1])
        missing = forgetting([[.79, .5], [.3, .2]])
        self.assertIsNone(missing['mean'])
        self.assertEqual(missing['status'], '不适用')
        self.assertEqual(missing['cells_reached_80'], 0)
        self.assertEqual(missing['cells_drop_over_20pp'], 0)
    def test_boundary(self):
        c=cells();v=np.zeros(42)
        cell=next(x for x in c if x['region']==3 and x['distance_band']==1 and x['azimuth_sector']==3 and x['height_stratum']==2)
        v[cell['cell_id']]=.5;b=boundary(v,c)
        self.assertEqual(b[-1]['upper_radius_m'],3.);self.assertIsNone(b[0]['upper_radius_m'])
    def test_coordination_and_failures(self):
        def r(success,height,target,reason=None,angle=.5):return dict(region=0,success=success,failure_reason=reason,
            minimum_base_height_m=height,target_height_m=target,azimuth_sector=3,terminal_heading_error_rad=angle)
        d=physical_and_coordination([r(True,1,2),r(True,2,4,angle=1),r(False,0,0,'tip')])
        self.assertAlmostEqual(d['physical_failure_rates']['near']['tip'],1/3)
        self.assertAlmostEqual(d['coordination']['minimum_body_target_height_correlation'],1)
        self.assertEqual(d['coordination']['back_success_heading_errors_rad'],[.5,1.])
        self.assertIsNone(physical_and_coordination([])['coordination']['minimum_body_target_height_correlation'])
    def test_coverage_auc_endpoint_and_comparison(self):
        c=cells();regions=np.array([x['region'] for x in c]);pool=dict(task_id=np.arange(42),cell_id=np.arange(42),region=regions)
        ev=[]
        for update,n in [(0,0),(100,21),(200,42),(300,42)]:
            rows=[dict(task_id=i,success=i<n,region=int(regions[i]),failure_reason=None if i<n else 'timeout',
                minimum_base_height_m=.2+i/100,target_height_m=.1+i/100,
                azimuth_sector=c[i]['azimuth_sector'],terminal_heading_error_rad=.5) for i in range(42)]
            ev.append((update,rows))
        d=summarize(ev,pool,c,300)
        self.assertEqual([x['coverage'] for x in d['curve']],[0,.5,1,1])
        self.assertAlmostEqual(d['endpoint_coverage'],5/6);self.assertAlmostEqual(d['coverage_auc'],2/3)
        self.assertAlmostEqual(d['endpoint_regional_rates']['walk'], (.125 + 1 + 1) / 3)
        partial = summarize(ev[:2], pool, c, 300)
        self.assertIsNone(partial['endpoint_coverage'])
        self.assertIsNone(partial['endpoint_regional_rates']['walk'])
        self.assertIsNone(compare(partial, d)['endpoint_coverage']['manual_minus_uniform'])
        e=compare(d,d);self.assertEqual(e['endpoint_coverage']['manual_minus_uniform'],0)
        self.assertIsNone(e['arrival_time_ratios_uniform_over_manual']['walk'])
        with self.assertRaises(ValueError):curve([(100,ev[1][1][:-1])],pool)

if __name__=='__main__':unittest.main()
