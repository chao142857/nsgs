"""Restricted admissions for the post-seed43 P3S1 supplements."""
from pathlib import Path

from tools.build_p3b_pool import require, sha256
from tools.p2b_K2_protocol import config_for, read
from tools.p2b_formal_protocol import EVALUATION, TRAINING
from tools.p3b_protocol import HOSTS, actual_gpu, checked, settings

MACHINE = 'yanglab'
PARENT = 'e2425b843b2369498cea0b540d0e1393f90fce9feb81f2786c11630e5a531167'
LOCK = 'bf28191e97855019ec5cdb20de1bfc4e6b08dd8ce70cd7104b4e5c5684a8fb67'
DECISION = 'fcd1c0a0870e6f5090d8a9233b8d99212a1a277ae04394f28846c0972a29c744'
ACTOR = '85664b37f007e5f6f4a819d931c58c483b6dce7b6f68afdb6aa3fb5124c02330'
MID = ('table_sharp', 'neural_sharp')
GRID = ('grid16_sharp', 'grid32_sharp')
OLD = '/data/nsgs_goalreach_p2b_formal_20261114'
OLD_BRIDGE = 'p2b-K2-verify-natural-R0-sgs_sharp-20-seed43-u20'
OLD_FORMAL = 'p2b-formal-natural-R0-sgs_sharp-seed43-u8000'


def job(scope, arm):
    seed = 42 if scope == 'engineering' else 43
    updates = {'bridge': 20, 'engineering': 200, 'formal': 8000}[scope]
    return dict(scope=scope, arm=arm, seed=seed, updates=updates,
                output=f'p3s1-{scope}-yanglab-{arm}-seed{seed}-u{updates}',
                num_envs=4096, rollout_steps=24, from_scratch=True)


def schedule(part):
    if part == 'middle':
        return dict(bridge=[job('bridge', MID[0])],
                    engineering=[job('engineering', MID[1])],
                    formal_if_reused=[job('formal', MID[1])],
                    formal_if_not_reused=[job('formal', arm) for arm in MID])
    require(part == 'grid', 'Unknown P3S1 part')
    return {scope: [job(scope, arm) for arm in GRID] for scope in ('engineering', 'formal')}


def pool_for(part, parent_pool):
    if part == 'middle':
        training = dict(TRAINING['natural'], size=32768)
        evaluation = dict(EVALUATION, size=2016)
        return dict(training=training, evaluation=evaluation)
    return {role: parent_pool[role] for role in ('training', 'evaluation')}


def load(project, digest):
    deployment = checked(project/'delivery/deployment.json', digest)
    part = deployment['part']
    require(deployment['revision']=='P3S1-v1' and deployment['schedule']==schedule(part)
            and deployment['host']==HOSTS[MACHINE] and deployment['decision_sha256']==DECISION
            and deployment['parent_sha256']==PARENT and deployment['selection_sha256']==LOCK,
            'P3S1 scope, schedule or parent binding differs')
    protocol = checked(project/'delivery/protocol.json', deployment['protocol_sha256'])
    require(protocol['part']==part and protocol['decision_sha256']==DECISION
            and protocol['parent_parameter_lock_sha256']==LOCK, 'P3S1 protocol differs')
    require(sha256(project/'delivery/decision.md')==DECISION, 'Decision file differs')
    for name, expected in deployment['files_sha256'].items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and sha256(project/path)==expected,
                'P3S1 frozen file differs: '+name)
    parent = checked(project/'delivery/p3b_P3B1_deployment.json', PARENT)
    lock = checked(project/'delivery/p3b_selection_v1.json', LOCK)
    original = checked(project/'delivery/p3b_P3B1_pool_manifest.json', parent['pool_manifest_sha256'])
    pool = pool_for(part, original)
    require(pool==deployment['pool'] and settings(lock)==deployment['scorer_settings'],
            'P3S1 pool or scorer settings differ')
    for item in pool.values():
        require(sha256(project/item['path'])==item['file_sha256'], 'P3S1 pool file differs: '+item['path'])
    if part=='middle':
        require(deployment['feature_binding']==checked(project/'delivery/features.json',
            deployment['feature_binding_sha256']), '32,768 feature manifest differs')
    else:
        preflight = read(project/'delivery/grid_preflight.json')
        require(deployment['feature_binding']==parent['feature_binding'] and preflight['passed'] is True
                and preflight['pool_sha256']==pool['training']['file_sha256']
                and preflight['implementation_sha256']==sha256(project/'goalreach/p3s1_grid.py'),
                'Parent feature or grid geometry binding differs')
    return deployment


def formal_permit(project, deployment, deploy_sha, permit_sha):
    permit = checked(project/'delivery/permit.json', permit_sha)
    acceptance = checked(project/'delivery/acceptance.json', permit['acceptance_sha256'])
    part = deployment['part']
    require(permit['part']==part and acceptance['part']==part
            and permit['deployment_sha256']==acceptance['deployment_sha256']==deploy_sha
            and acceptance['passed'] is True and permit['expected_initial_actor_sha256']==ACTOR,
            'P3S1 permit or acceptance differs')
    key = 'formal'
    if part=='middle':
        baseline = checked(project/'delivery/baseline_reuse.json', permit['baseline_reuse_sha256'])
        require(type(baseline['reuse']) is bool and baseline['deployment_sha256']==deploy_sha,
                'P3S1 baseline decision differs')
        key = 'formal_if_reused' if baseline['reuse'] else 'formal_if_not_reused'
    require(permit['jobs']==deployment['schedule'][key], 'P3S1 formal jobs differ')
    return permit


def admission(project, args):
    scope, arm, digest = args.p3b_scope, args.p3b_arm, args.p3b_deployment_sha256
    require(scope in ('bridge','engineering','formal') and args.p3b_machine==MACHINE
            and args.headless and args.device=='cuda:0' and args.num_envs==4096 and args.throttle==0
            and bool(args.p3b_permit_sha256)==(scope=='formal'), 'Invalid P3S1 entry')
    deploy = load(project, digest)
    host = actual_gpu(MACHINE)
    permit = formal_permit(project,deploy,digest,args.p3b_permit_sha256) if scope=='formal' else None
    authorized = permit['jobs'] if permit else deploy['schedule'].get(scope, ())
    item = job(scope,arm)
    require(item in authorized,'P3S1 job is outside the frozen schedule')
    output = project/'runs'/item['output']
    require(args.out.absolute()==args.out.resolve()==output and not output.exists()
            and not output.is_symlink() and not output.parent.is_symlink(),'Invalid P3S1 output')
    config = config_for('natural','sgs_sharp',item['seed'],MACHINE)
    config.update(phase='P3S1',protocol_revision='P3S1-v1',arm_id=arm,launch_status=scope,
        updates=item['updates'],sampler='external' if arm=='neural_sharp' else 'sgs',output_label=item['output'],
        raw_evaluation_every_updates=200 if scope=='engineering' else 500,
        checkpoint_every_updates=200 if scope=='engineering' else 500)
    for role,item in deploy['pool'].items():
        config.update({role+'_pool':item['path'],role+'_pool_size':item['size'],
                       role+'_pool_content_sha256':item['content_sha256'],role+'_pool_file_sha256':item['file_sha256']})
    bins = {'grid16_sharp':16, 'grid32_sharp':32}.get(arm)
    return dict(deploy,host=host,permit=permit,machine=MACHINE,grid_bins=bins),config


def verify_initial(deployment, config, digest, out):
    if config['student_seed']==43:
        require(digest==ACTOR, 'P3S1 seed43 did not reproduce the initial actor')
    permit = deployment['permit']
    if permit and config['arm_id']=='neural_sharp' and permit['jobs'][0]['arm']=='table_sharp':
        peer = out.parent/job('formal','table_sharp')['output']
        result, audit = read(peer/'result.json'), read(peer/'audit.json')
        require(result['completed_updates']==8000 and audit['passed'] is True
                and audit['training_actor_sha256']==digest, 'P3S1 new table peer is absent or differs')
