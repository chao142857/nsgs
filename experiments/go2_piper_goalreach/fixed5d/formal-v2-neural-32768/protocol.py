"""Frozen physical coordinates, draw streams, and sparse evaluation schedule."""
from hashlib import sha256
import json


VERSION = 'episode-fixed-reach-5d-mu040-100-slope21-r1'
WIDE = True
MU_BOUNDS = (0.40, 1.00)
ALPHA_BOUNDS_DEG = (-21.0, 21.0)
TARGET_CLEARANCE_M = 1.e-6
PLATFORM_METRES = (9.5, 9.5, .4)
DRAW_KEYS = {'train': 2026112001, 'development': 2026112002, 'test': 2026112003}
AXES = ('horizontal_distance', 'azimuth', 'height', 'mu', 'alpha_deg')
BOUNDS = ((0., 3.), (-3.141592653589793, 3.141592653589793), (.1, .85),
          MU_BOUNDS, ALPHA_BOUNDS_DEG)
METHODS = ('nsgs_e4_sharp',)
LIGHT = tuple(range(500, 8000, 1000))
DUAL = tuple(range(1000, 8001, 1000))
FINAL_EMA = (7800, 7900)
FULL = frozenset((*DUAL, *FINAL_EMA))
DEVELOPMENT = tuple(sorted((*LIGHT, *FULL)))
TEST_ROWS = 8064
TOTAL_ROWS = len(LIGHT)*504 + len(DUAL)*2*4032 + len(FINAL_EMA)*4032 + TEST_ROWS


def file_sha(path):
    h = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def modes(update, sealed=False):
    assert update in DEVELOPMENT
    if sealed:
        assert update == 8000
    return ('raw', 'ema099') if update in DUAL and not sealed else ('ema099',)


def interval_auc(curve):
    assert [row['update'] for row in curve] == list(DEVELOPMENT)
    return sum((b['update']-a['update'])*(a['coverage']+b['coverage'])/2
               for a,b in zip(curve,curve[1:]))/7500


def schedule(panel_sha):
    return dict(version=VERSION, development_nodes=list(DEVELOPMENT), ema_504=list(LIGHT),
                development_full_raw_and_ema=list(DUAL), development_full_ema_only=list(FINAL_EMA),
                ordinary_hundred_evaluation=False, ordinary_hundred_discard=False,
                terminal_update=8000, terminal_rows=TEST_ROWS, total_evaluation_rows=TOTAL_ROWS,
                auc_interval=[500,8000], auc_normalization_updates=7500, panel_sha256=panel_sha,
                from_initial_weights=True)


def validate(directory):
    panel = directory/'development_504.npz'
    freeze = json.loads((directory/'pool_manifest.json').read_bytes())
    panel_sha = file_sha(panel)
    mapping = directory/'original_index_mapping.json'
    assert panel_sha == freeze['panels']['development_504']['sha256']
    assert file_sha(mapping) == freeze['mapping_sha256']
    schedule_file = directory/'sparse_schedule.json'
    assert json.loads(schedule_file.read_bytes()) == schedule(panel_sha)
    return panel, json.loads(mapping.read_bytes()), file_sha(schedule_file)
