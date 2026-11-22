"""Track the same 504 frozen physical configurations at every development node."""
import json

from protocol import file_sha


ROW_KEYS = ('task_id','target_index','config_id','mu','alpha_deg','cell_id','region')


def coverage_point(out, update, match, *, compact):
    stamp = f'{update:04d}'
    stage = out/'evaluations'
    proof = json.loads((stage/f'{stamp}_switch_audit.json').read_bytes())
    path = stage/f'{stamp}_ema099/rows.json'
    digest = file_sha(path)
    assert proof['row_sources']['ema099'] == digest and proof['update'] == update
    rows = json.loads(path.read_bytes())
    assert len(rows) == proof['rows']['ema099'] == (504 if compact else 4032)
    assert [row['task_id'] for row in rows] == list(range(len(rows)))
    successes = [0]*42
    for entry in match:
        row = rows[entry['task_id'] if compact else entry['original_task_id']]
        assert row['target_index'] == entry['target_index' if compact else 'original_target_index']
        assert all(row[name] == entry[name] for name in ('config_id','mu','alpha_deg','cell_id'))
        assert row['success'] in (0,1,False,True)
        successes[row['cell_id']] += int(row['success'])
    return dict(update=update,coverage=sum(n/12 for n in successes)/42,cell_successes=successes,
                panel='compact' if compact else 'subset_from_full',source_rows_sha256=digest,
                wall_s=proof['wall_s'])
