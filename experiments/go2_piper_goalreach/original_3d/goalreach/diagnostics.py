"""Statistics shared by noise scans, training windows and fixed evaluations."""
from collections import Counter
import numpy as np
import torch

PHYSICAL=('nonfoot_contact','base_contact','tip')
BANDS=('0-2cm','2-5cm','5-10cm')


def gaussian_action(template,sigma,scope='all'):
    if sigma<0 or scope not in ('all','legs','arm'):raise ValueError((sigma,scope))
    a=torch.randn_like(template)*sigma
    if scope=='legs':a[:,12:]=0
    if scope=='arm':a[:,:12]=0
    return a


def episode_summary(rows, compare_hold_speed=False):
    n=len(rows);counts=Counter('success' if r['success'] else r['failure_reason'] for r in rows)
    wins=[r['episode_steps'] for r in rows if r['success']]
    return dict(n=n,counts=dict(counts),success_rate=counts['success']/n if n else None,
        rates={reason:counts[reason]/n if n else None for reason in ('nonfoot_contact','base_contact','tip','timeout')},
        physical_failure_rate=sum(counts[r] for r in PHYSICAL)/n if n else None,
        mean_episode_steps=float(np.mean([r['episode_steps'] for r in rows])) if n else None,
        successful_steps=dict(n=len(wins),histogram=dict(sorted(Counter(map(str,wins)).items())),
            min=min(wins) if wins else None,median=float(np.median(wins)) if wins else None,
            p90=float(np.percentile(wins,90)) if wins else None,max=max(wins) if wins else None),
        distance_bands={label:episode_summary_flat([r for r in rows if r.get('near_distance_band')==i], compare_hold_speed) for i,label in enumerate(BANDS)},
        **(hold_speed_comparison(rows) if compare_hold_speed else {}))


def hold_speed_comparison(rows):
    successful = [row for row in rows if row['success']]
    measured = [row for row in successful if 'hold_speed_max_m_s' in row]
    assert len(measured) == len(successful)
    low = sum(row['hold_speed_max_m_s'] < .25 for row in measured)
    return dict(success_hold_speed_lt_025=dict(n=len(measured), count=low, fraction=low/len(measured) if measured else None))


def episode_summary_flat(rows, compare_hold_speed=False):
    # The leaf intentionally has no nested distance bands.
    n=len(rows);counts=Counter('success' if r['success'] else r['failure_reason'] for r in rows)
    wins=[r['episode_steps'] for r in rows if r['success']]
    return dict(n=n,counts=dict(counts),success_rate=counts['success']/n if n else None,
        rates={reason:counts[reason]/n if n else None for reason in ('nonfoot_contact','base_contact','tip','timeout')},
        physical_failure_rate=sum(counts[r] for r in PHYSICAL)/n if n else None,
        mean_episode_steps=float(np.mean([r['episode_steps'] for r in rows])) if n else None,
        successful_steps=dict(n=len(wins),histogram=dict(sorted(Counter(map(str,wins)).items())),
            min=min(wins) if wins else None,median=float(np.median(wins)) if wins else None,
            p90=float(np.percentile(wins,90)) if wins else None,max=max(wins) if wins else None),
        **(hold_speed_comparison(rows) if compare_hold_speed else {}))


def classify(evaluations):
    """Frozen before Phase 1: EMA, final three means; trend over 1000..1500."""
    by={x['update']:x for x in evaluations}
    required=[1000,1100,1200,1300,1400,1500]
    if any(i not in by for i in required):return dict(category='未完成',reason='Missing required evaluations')
    final=[by[i] for i in [1300,1400,1500]]
    overall=float(np.mean([r['success_rate'] for r in final]))
    far=float(np.mean([r['distance_bands']['5-10cm']['success_rate'] for r in final]))
    physical=float(np.mean([r['physical_failure_rate'] for r in final]))
    values=[by[i]['success_rate'] for i in required]
    slope=float(np.polyfit(np.arange(6),values,1)[0])
    rising=slope>1e-12 and np.mean(values[-3:])>np.mean(values[:3])+1e-12
    category='通过' if overall>=.8 and far>=.6 and physical<=.05 else '有进展但未完成' if overall>=.55 and far>=.3 and rising else '不通过'
    return dict(category=category,final_three_overall=overall,final_three_far=far,
        final_three_physical=physical,last500_slope_per100_updates=slope,last500_rising=bool(rising),
        trend_definition='EMA near overall at 1000..1500: positive OLS slope AND final-three mean greater than first-three mean')
