"""Offline metrics, never used by training or curriculum decisions."""
import numpy as np
from .core import REGIONS


def curve(evaluations,pool):
    """Input [(update, rows)]; strict complete, exactly-once task coverage."""
    result=[]
    for update,rows in sorted(evaluations):
        ids=[r['task_id'] for r in rows]
        if sorted(ids)!=list(range(len(pool['task_id']))):
            raise ValueError('Evaluation must contain every task exactly once')
        by_id={r['task_id']:r for r in rows}
        rates=[];regions=[]
        for cell in range(42):
            ix=np.flatnonzero(pool['cell_id']==cell)
            if not len(ix):raise ValueError('Missing evaluation cell')
            rates.append(float(np.mean([by_id[int(i)]['success'] for i in ix])))
            regions.append(int(pool['region'][ix[0]]))
        result.append(dict(update=int(update),coverage=float(np.mean(rates)),cell_rates=rates,
            regions={name:float(np.mean(np.asarray(rates)[np.array(regions)==i])) for i,name in enumerate(REGIONS)}))
    return result


def arrival(updates, rates):
    for i in range(len(rates) - 2):
        if all(rate >= .5 for rate in rates[i:i + 3]):
            return int(updates[i])
    return None


def forgetting(cell_rates):
    values = []
    for rates in np.asarray(cell_rates).T:
        reached = np.flatnonzero(rates >= .8)
        if not len(reached):
            values.append(None)
            continue
        tail = rates[reached[0]:]
        values.append(float(np.max(np.maximum.accumulate(tail) - tail)))
    eligible = [value for value in values if value is not None]
    return dict(
        per_cell=values,
        mean=float(np.mean(eligible)) if eligible else None,
        status='applicable' if eligible else '不适用',
        cells_reached_80=len(eligible),
        cells_drop_over_20pp=sum(value > .2 + 1e-12 for value in eligible),
        never_reached_80=len(values) - len(eligible),
    )


def boundary(cell_rates,cells):
    """Only cells with both azimuth and height strata have identifiable radii."""
    rows=[]
    for az in range(4):
        for h in range(3):
            passing=[c for c in cells if c['region']>=2 and c['azimuth_sector']==az
                     and c['height_stratum']==h and cell_rates[c['cell_id']]>=.5]
            edge=lambda c:1.5 if c['region']==2 else 2.25 if c['distance_band']==0 else 3.
            far=max(passing,key=edge) if passing else None
            rows.append(dict(azimuth_sector=az,height_stratum=h,
                furthest_distance_band=None if far is None else ('step' if far['region']==2 else ['walk_1.5_2.25','walk_2.25_3'][far['distance_band']]),
                upper_radius_m=None if far is None else edge(far)))
    return rows


def physical_and_coordination(rows):
    physical={}
    for i,name in enumerate(REGIONS):
        rr=[r for r in rows if r['region']==i]
        physical[name]={reason:sum(r['failure_reason']==reason for r in rr)/len(rr) if rr else None
            for reason in ('nonfoot_contact','base_contact','tip')}
        physical[name]['any']=sum(r['failure_reason'] in ('nonfoot_contact','base_contact','tip') for r in rr)/len(rr) if rr else None
    successful=[r for r in rows if r['success']]
    x=np.array([r['minimum_base_height_m'] for r in successful]);y=np.array([r['target_height_m'] for r in successful])
    correlation=float(np.corrcoef(x,y)[0,1]) if len(x)>1 and np.ptp(x)>0 and np.ptp(y)>0 else None
    angles=[float(r['terminal_heading_error_rad']) for r in successful if r['azimuth_sector']==3]
    return dict(physical_failure_rates=physical,coordination=dict(successful_episodes=len(successful),
        minimum_body_target_height_correlation=correlation,back_success_heading_errors_rad=angles,
        back_heading_quantiles_rad=np.quantile(angles,[0,.25,.5,.75,1]).tolist() if angles else None))


def summarize(evaluations, pool, cells, total_updates):
    c=curve(evaluations,pool)
    if not c:raise ValueError('No evaluations')
    times=[r['update'] for r in c];coverage=[r['coverage'] for r in c]
    if total_updates<=0 or max(times)>total_updates:raise ValueError('Invalid total update budget')
    endpoint = c[-3:]
    has_endpoint = len(endpoint) == 3
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    result=dict(curve=c,endpoint_coverage=float(np.mean(coverage[-3:])) if has_endpoint else None,
        endpoint_evaluations=len(endpoint),
        endpoint_regional_rates={r:float(np.mean([x['regions'][r] for x in endpoint])) if has_endpoint else None
                                 for r in REGIONS},
        coverage_auc=float(integrate(coverage,times)/total_updates),auc_observed_interval=[min(times),max(times)],
        auc_denominator=total_updates,auc_extrapolation='none',
        arrival_times={r:arrival(times,[x['regions'][r] for x in c]) for r in REGIONS},
        forgetting=forgetting([x['cell_rates'] for x in c]),
        boundaries=[dict(update=x['update'],sectors=boundary(x['cell_rates'],cells)) for x in c],
        evaluation_diagnostics=[dict(update=int(u),**physical_and_coordination(rows)) for u,rows in sorted(evaluations)])
    return result


def compare(uniform, manual):
    def rates(uniform_rate, manual_rate):
        difference = None
        if uniform_rate is not None and manual_rate is not None:
            difference = manual_rate - uniform_rate
        return dict(uniform=uniform_rate, manual=manual_rate, manual_minus_uniform=difference)

    def ratio(a, b):
        return None if a is None or b in (None, 0) else a / b

    return dict(
        endpoint_coverage=rates(uniform['endpoint_coverage'], manual['endpoint_coverage']),
        endpoint_regions={r:rates(uniform['endpoint_regional_rates'][r], manual['endpoint_regional_rates'][r])
                          for r in ('step', 'walk')},
        arrival_time_ratios_uniform_over_manual={r:ratio(uniform['arrival_times'][r], manual['arrival_times'][r])
                                                 for r in REGIONS},
        missing_ratio='null when either arm never arrived or denominator is zero; inspect per-arm arrival times',
    )
