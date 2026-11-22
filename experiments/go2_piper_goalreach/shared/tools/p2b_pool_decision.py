"""Compute the locked main pool from completed Phase 2A evidence on Mac."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET = 8000
TAIL = (7800, 7900, 8000)
EMA_UPDATES = list(range(100, 8001, 100))
RAW_UPDATES = list(range(500, 8001, 500))
EFFECTIVE = dict(reward='R0', student_seed=42, initial_action_std=.3, entropy_coef=.001,
                 gamma=.99, success_criterion='hold_only')
RULE = dict(a4_minimum_walk=0.50, minimum_advantage=0.20, minimum_arrival_ratio=1.5,
            preferred_if_both_qualify='natural', use_pilot_results=False,
            missing_or_failed_natural_evidence='pending',
            arrival='First of three consecutive complete EMA walk rates >= 0.5, else null')


def read(path):
    return json.loads(path.read_bytes())


def source(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fraction(point):
    wins, tasks = point['successes'], point['tasks']
    require(type(wins) is int and tasks == 1008 and 0 <= wins <= tasks, 'Invalid walk count')
    return Fraction(wins, tasks)


def metric(arm, claim, summary, queue, pool, evaluation):
    require(queue['phase'] == 'complete' and queue['completed_updates'] == BUDGET and queue['ended_unix'] > queue['started_unix'],
            arm+' is missing a normal completion receipt')
    require(claim['passed'] is True and claim['finite'] is True and claim['exit_code'] == 0,
            arm+' did not pass the runner and launcher')
    require(claim['completed_updates'] == BUDGET and claim['num_envs'] == 4096 and claim['optimizer_calls'] == 160000,
            arm+' has inconsistent completion accounting')
    require(claim['ema_updates'] == EMA_UPDATES and claim['raw_updates'] == RAW_UPDATES,
            arm+' has incomplete evaluation cadence')
    require(claim['checkpoints'] == [f'checkpoint_{i:04d}.pt' for i in RAW_UPDATES],
            arm+' has an incomplete checkpoint list')
    require(claim['execution_settings'] == summary['execution_settings'] == EFFECTIVE,
            arm+' has mismatching actual settings')
    protocol = claim['protocol']
    require(protocol == summary['protocol'] and summary['passed'] and summary['completed_updates'] == BUDGET,
            arm+' report does not match the confirmed run')
    expected = dict(phase='2A', arm=arm, num_envs=4096, student_seed=42, rollout_steps=24, updates=8000,
                    timeout_penalty=-50., from_scratch=True, ema_decay=.99, evaluation_pool_size=2016,
                    evaluation_every_updates=100, raw_evaluation_every_updates=500, checkpoint_every_updates=500,
                    curriculum=dict(success_threshold=.6, window_updates=50, minimum_episodes=200, newest_mass=.5),
                    sampler='uniform' if arm in ('A1','A3') else 'manual',
                    training_pool=pool['path'], evaluation_pool=evaluation['path'],
                    training_pool_content_sha256=pool['content_sha256'],
                    evaluation_pool_content_sha256=evaluation['content_sha256'])
    require(all(protocol.get(k) == v for k,v in expected.items()), arm+' protocol or content hash mismatch')
    require(claim['training_file_sha256'] == pool['file_sha256']
            and claim['evaluation_file_sha256'] == evaluation['file_sha256'], arm+' pool file hash mismatch')
    for field in ('result','launcher_result','episode_manifest'):
        require(len(claim[field]['sha256']) == 64 and claim[field]['path'], arm+' missing source provenance')
    points = claim['ema_tail']
    require(tuple(p['update'] for p in points) == TAIL, arm+' missing tail counts')
    rates = [fraction(p) for p in points]
    curve = summary['ema']['curve']
    require([v['update'] for v in curve] == EMA_UPDATES, arm+' has incomplete EMA summary')
    require(all(abs(v['regions']['walk'] - float(w)) < 1e-12 for v,w in zip(curve[-3:], rates)),
            arm+' counts do not match walk rates')
    arrival = next((a['update'] for a,b,c in zip(curve,curve[1:],curve[2:])
                    if all(x['regions']['walk'] >= .5 for x in (a,b,c))), None)
    require(arrival == summary['ema']['arrival_times']['walk'], arm+' arrival does not match the EMA series')
    mean = sum(rates, Fraction(0))/3
    return dict(mean=float(mean), arrival=arrival, ema_tail=points,
                mean_numerator=mean.numerator, mean_denominator=mean.denominator,
                execution_settings=claim['execution_settings'], run=claim['run'])


def pair(manual, uniform):
    a = Fraction(manual['mean_numerator'], manual['mean_denominator'])
    b = Fraction(uniform['mean_numerator'], uniform['mean_denominator'])
    advantage = a-b
    ratio = Fraction(uniform['arrival'], manual['arrival']) if uniform['arrival'] and manual['arrival'] else None
    return dict(advantage_pp=float(advantage*100), arrival_ratio=float(ratio) if ratio is not None else None,
                qualified=a >= Fraction(1,2) and (advantage >= Fraction(1,5) or
                      (ratio is not None and ratio >= Fraction(3,2))))


def choose(balanced, a4, a3=None):
    if not balanced['qualified']:
        return 'pending', None, 'Balanced evidence has not met the paired gate'
    if a4 is None:
        return 'pending', None, 'Awaiting normal A4 completion with the three raw EMA counts'
    if Fraction(a4['mean_numerator'], a4['mean_denominator']) < Fraction(1,2):
        return 'locked', 'balanced', 'A4 completed with mean walk success below 50%; A3 was not triggered'
    if a3 is None:
        return 'pending', None, 'A4 met 50%; awaiting complete same-host A3 evidence'
    if pair(a4, a3)['qualified']:
        return 'locked', 'natural', 'Natural pair passed; natural has priority among qualified pools'
    return 'locked', 'balanced', 'Natural pair completed without meeting the full gate'


def natural_completion(summary, queue, gate, inventory, arm, metadata, evaluation):
    # The 2A controller marks complete only after validating runner/launcher
    # exit, every task ID, both cadences and the full checkpoint set.
    entries = {item['path']: item for item in inventory['files']}
    run = Path(next(item for item in entries if item.endswith(f"/runs/{queue['label']}/result.json"))).parent
    job = run.parent.parent/'jobs'/run.name
    lookup = lambda path: dict(path=str(path), sha256=entries[str(path)]['sha256'])
    evals = run/'evaluations'
    points = []
    for row in summary['ema']['curve'][-3:]:
        update = row['update']
        # Walk spans 24 equal-sized cells at IDs 18..41. The full-set
        # summary is computed from verified complete per-task rows.
        walk = row['cell_rates'][18:]
        counts = [round(rate*42) for rate in walk]
        require(len(counts) == 24 and all(abs(rate - n/42) < 1e-12 for rate,n in zip(walk,counts)),
                arm+' walk cell counts are not reconstructible')
        points.append(dict(update=update, successes=sum(counts), tasks=1008,
                           rows=lookup(evals/f'{update:04d}_ema099/rows.json'),
                           switch_audit=lookup(evals/f'{update:04d}_switch_audit.json'),
                           count_provenance='original A4 gate' if arm == 'A4' else 'integer counts from complete 42-task cells'))
    if arm == 'A4':
        require([(x['update'],x['successes'],x['tasks']) for x in gate['evaluations']] ==
                [(x['update'],x['successes'],x['tasks']) for x in points], 'Published A4 raw counts differ')
    return dict(passed=summary['passed'], finite=True, exit_code=0, completed_updates=summary['completed_updates'],
                num_envs=summary['protocol']['num_envs'], optimizer_calls=BUDGET*20,
                execution_settings=summary['execution_settings'], protocol=summary['protocol'],
                training_file_sha256=metadata['file_sha256'], evaluation_file_sha256=evaluation['file_sha256'],
                ema_updates=[i for i in EMA_UPDATES if str(evals/f'{i:04d}_ema099/rows.json') in entries],
                raw_updates=[i for i in RAW_UPDATES if str(evals/f'{i:04d}_raw/rows.json') in entries],
                checkpoints=sorted(Path(name).name for name in entries if name.startswith(str(run)) and name.endswith('.pt')),
                ema_tail=points, result=lookup(run/'result.json'), launcher_result=lookup(job/'result.json'),
                episode_manifest=lookup(run/'episodes/manifest.json'), run=str(run))


def collect(pool, metadata, evaluation):
    base = ROOT.parents[2]/'nsgs_goalreach_p2a_20261113/projects/goalreach'
    natural = pool == 'natural'
    summary_dir = base/('report_2a_R0_natural_8000' if natural else 'report_2a_R0_8000')
    report = base/('REPORT_2A_R0_NATURAL_8000.md' if natural else 'REPORT_2A_R0_8000.md')
    status_file = summary_dir/'status.json'
    completion_path = (summary_dir/'completion_evidence.json' if natural else ROOT/'delivery/2a_balanced_completion.json')
    inputs = {name:source(path) for name,path in {
        'report':report, 'status':status_file, 'completion_evidence':completion_path}.items() if path.exists()}
    evidence = dict(inputs=inputs,runs={})
    required = (report,status_file) if natural else (report,status_file,completion_path)
    if not all(path.exists() for path in required):
        evidence['pending_reason'] = 'Awaiting the Mac report and its auditable completion receipt'
        return evidence
    status = read(status_file)
    host = 'ubuntu-robot' if natural else 'yanglab'
    proof = read(completion_path) if completion_path.exists() else dict(host=host,runs={})
    require(status['budget'] == BUDGET and proof['host'] == host and status.get('host', host) == host,
            'Task host or budget differs')
    evidence['code_version'] = status['code_version']
    if natural:
        for key, binding in (('train',metadata),('evaluation',evaluation)):
            for field in ('content_sha256','file_sha256'):
                require(status['pools'][key][field] == binding[field], 'Natural pool provenance differs')
    else:
        require(status['phase'] == 'complete', 'Balanced queue did not complete normally')
    for arm in (('A4','A3') if natural else ('A2','A1')):
        queue = status['arms'][arm]
        if queue['phase'] != 'complete':
            if not natural:
                raise ValueError('Balanced arm incomplete: '+arm)
            if queue['phase'] == 'stopped':
                evidence['pending_reason'] = arm+' stopped: '+str(queue.get('error'))
            else:
                evidence['pending_reason'] = arm+' status '+queue['phase']
            break
        path = summary_dir/arm/'summary.json'
        if not path.exists():
            evidence['pending_reason'] = arm+' summary missing'
            break
        summary = read(path)
        claim = proof['runs'].get(arm)
        if natural and claim is None:
            inventory = summary_dir/arm/'retained_files.json'
            if not inventory.exists():
                evidence['pending_reason'] = arm+' retained-file receipt missing'
                break
            evidence['inputs'][arm+'_retained_files'] = source(inventory)
            claim = natural_completion(summary, queue, status['gate'], read(inventory), arm, metadata, evaluation)
        if claim is None:
            raise ValueError('Missing completion receipt for '+arm)
        evidence['inputs'][arm+'_summary'] = source(path)
        evidence['runs'][arm] = metric(arm, claim, summary, queue, metadata, evaluation)
        if arm in ('A2','A4'):
            expected = status['gate']['evaluations']
            actual = evidence['runs'][arm]['ema_tail']
            require([(x['update'],x['successes'],x['tasks']) for x in expected] ==
                    [(x['update'],x['successes'],x['tasks']) for x in actual], 'A2/A4 gate counts disagree')
            require(abs(status['gate']['mean'] - evidence['runs'][arm]['mean']) < 1e-12,
                    'A2/A4 gate mean disagrees')
    if len(evidence['runs']) == 2:
        first, second = (('A4','A3') if natural else ('A2','A1'))
        evidence['pair'] = pair(evidence['runs'][first], evidence['runs'][second])
    return evidence


def build():
    # Import lazily so the evidence decision can be composed using the Mac's Python.
    pool_file = ROOT/'delivery/p2b_candidate_pools.json'
    pools = read(pool_file)
    evidence = {}
    for pool in ('balanced','natural'):
        try:
            evidence[pool] = collect(pool, pools[pool], pools['evaluation'])
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
            evidence[pool] = dict(runs={}, pending_reason=f'{type(error).__name__}: {error}')
    natural = evidence['natural']
    status, main, reason = choose(evidence['balanced'].get('pair', dict(qualified=False)),
                                  natural['runs'].get('A4'), natural['runs'].get('A3'))
    if status == 'pending':
        reason += '. '+str(natural.get('pending_reason', evidence['balanced'].get('pending_reason','')))
    return dict(schema='goalreach-p2b-main-pool-v1', status=status, main_pool=main, reason=reason,
                rules=RULE, evidence=evidence, candidate_pool_evidence=source(pool_file))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    target = ROOT/'delivery/p2b_main_pool_decision.json'
    decision = build()
    if target.exists() and read(target)['status'] == 'locked':
        if read(target) != decision:
            raise SystemExit('Existing locked decision differs; preserve the published decision')
    else:
        temp = target.with_suffix('.tmp')
        temp.write_text(json.dumps(decision, indent=2)+'\n')
        temp.replace(target)
    print(decision['status'], decision['main_pool'], decision['reason'])
