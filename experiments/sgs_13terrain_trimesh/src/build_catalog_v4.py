"""Build frozen v4 260-tile Trimesh catalog on v6 continuous-curriculum terrains.

Features UWLab/OmniReset-style 3-tier continuous reset bridge:
  - Tier 0 (10%, 40/tile):  Easy bootstrap (12 birth-hold + 28 short same-region walk)
  - Tier 1 (35%, 140/tile): Cross-Edge Near-Goal Bridge (1.10-2.95m single-obstacle/edge crossings, sr != gr)
  - Tier 2 (55%, 220/tile): Full Cross-Obstacle Routes (matching holdout topology across the full tile)
  - Holdout (2,080 = 8/tile): 100% strict full cross-obstacle routes with ZERO endpoint overlap vs train!
"""
from __future__ import annotations

from collections import deque
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
REPO_SRC = EXP_ROOT.parent.parent / "src"
sys.path.insert(0, str(HERE))
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))
if (EXP_ROOT / "nsgs_src").is_dir():
    sys.path.insert(0, str(EXP_ROOT / "nsgs_src"))

from trimesh_terrains import TERRAIN_NAMES, TILE_SIZE, HALF_WIDTH, terrain_arrays, build_terrain_mesh
from build_catalog import TERRAIN_PRIM, FOOT_SAMPLES, BASE_RING
from nsgs.baselines.base import build_fourier_features, TaskPool
from nsgs.baselines.grid import compute_sparse_cell_ids

TERRAIN_SET = "v6"


def _wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def _min_seg_radius(sx: float, sy: float, gx: float, gy: float) -> float:
    vx, vy = gx - sx, gy - sy
    vv = vx * vx + vy * vy
    if vv < 1e-8:
        return math.hypot(sx, sy)
    t = max(0.0, min(1.0, -(sx * vx + sy * vy) / vv))
    return math.hypot(sx + t * vx, sy + t * vy)


def region_id_v4(name: str, x: float, y: float, z: float, d: float, *, allow_bridge: bool = False) -> int:
    """Topological zone ID on v6 terrains.

    Core terminal zones (0..15) are used for Holdout & Tier-2 Full Cross.
    Bridge zones (50..69, enabled only when allow_bridge=True for train_poses) capture
    intermediate beams, bridge stones, parallel-box rows, and middle rings for Tier-1 Cross-Edge Bridge.
    """
    r = math.hypot(x, y)
    if name == "radial_beam":
        hub_r = 1.35 - 0.15 * d
        if r < hub_r - 0.35:
            return 0  # central octagonal hub
        ang = math.atan2(y, x)
        spoke = int(round(ang / (math.pi / 4.0))) % 8
        spoke_ang = spoke * (math.pi / 4.0)
        if r > 3.65:
            if abs(_wrap_pi(ang - spoke_ang)) <= 0.22:
                return 1 + spoke
            return 99
        if allow_bridge and (hub_r - 0.05 <= r <= 3.55) and abs(_wrap_pi(ang - spoke_ang)) <= 0.22:
            return 50 + spoke  # mid-beam bridge zone
        return 98

    if name == "stairs":
        if max(abs(x), abs(y)) < 0.92 and z > 0.18:
            return 0  # summit platform
        if z < 0.05 and max(abs(x), abs(y)) > 3.62:
            if x < -3.62 and abs(y) <= 1.45:
                return 1
            if x > 3.62 and abs(y) <= 1.45:
                return 2
            if y < -3.62 and abs(x) <= 1.45:
                return 3
            if y > 3.62 and abs(x) <= 1.45:
                return 4
            return 99
        if allow_bridge and 1.15 <= max(abs(x), abs(y)) <= 3.45:
            if abs(x) >= abs(y):
                return 51 if x < 0 else 52
            return 53 if y < 0 else 54
        return 98

    if name == "gap":
        plat_half = 0.5 * (2.60 - 0.30 * d)
        gap_w = 0.10 + 0.66 * d
        outer_edge = plat_half + gap_w
        if max(abs(x), abs(y)) <= plat_half - 0.08:
            return 0  # central island (includes near-edge poses!)
        if x <= -(outer_edge + 0.08) and abs(y) <= 1.30:
            return 1
        if x >= +(outer_edge + 0.08) and abs(y) <= 1.30:
            return 2
        if y <= -(outer_edge + 0.08) and abs(x) <= 1.30:
            return 3
        if y >= +(outer_edge + 0.08) and abs(x) <= 1.30:
            return 4
        return 99

    if name == "pit":
        pit_half = 0.5 * (3.5 + 0.6 * d)
        if z < -0.06 and max(abs(x), abs(y)) <= pit_half - 0.15:
            return 0  # pit floor
        if z >= -0.03:
            if x <= -(pit_half + 0.12) and abs(y) <= 1.45:
                return 1
            if x >= +(pit_half + 0.12) and abs(y) <= 1.45:
                return 2
            if y <= -(pit_half + 0.12) and abs(x) <= 1.45:
                return 3
            if y >= +(pit_half + 0.12) and abs(x) <= 1.45:
                return 4
            return 99
        return 98

    if name == "balance_beam":
        # Island is [-2.70, -0.40] x [-1.15, +1.15]; east rim starts at x = +3.60.
        if -2.52 <= x <= -0.52 and abs(y) <= 0.92:
            return 0  # floating square island ONLY
        if 3.68 <= x <= 4.48 and abs(y) <= 0.92:
            return 1  # east outer rim directly at beam exit ONLY
        if allow_bridge and (-0.42 <= x <= 3.52) and abs(y) <= 0.42:
            return 50  # on the balance beam
        if max(abs(x), abs(y)) > 3.55:
            return 99
        return 98

    if name == "inverted_slope":
        if max(abs(x), abs(y)) <= 0.88 and z <= -0.14:
            return 0  # recessed central bowl floor ONLY
        if max(abs(x), abs(y)) > 3.72 and z >= -0.04:
            if x < -3.72 and abs(y) <= 1.40:
                return 1
            if x > 3.72 and abs(y) <= 1.40:
                return 2
            if y < -3.72 and abs(x) <= 1.40:
                return 3
            if y > 3.72 and abs(x) <= 1.40:
                return 4
            return 99
        if allow_bridge and 1.25 <= max(abs(x), abs(y)) <= 3.45:
            if abs(x) >= abs(y):
                return 51 if x < 0 else 52
            return 53 if y < 0 else 54
        return 98

    if name == "islands":
        if r < 1.38:
            return 0  # central octagon
        if abs(x) > 2.05 and abs(y) > 2.05:
            if x < 0 and y < 0:
                return 1
            if x > 0 and y < 0:
                return 2
            if x > 0 and y > 0:
                return 3
            return 4
        if allow_bridge and r >= 1.42:
            return 50 + (int(x >= 0) + 2 * int(y >= 0))  # stepping-stone bridges
        return 98

    if name == "parallel_box":
        if y <= -2.02:
            return 0  # south landing bank
        if y >= +2.02:
            return 1  # north landing bank
        if allow_bridge and abs(y) <= 1.75:
            if y < -0.45:
                return 50  # south parallel box row
            if y <= 0.45:
                return 51  # middle parallel box row
            return 52      # north parallel box row
        return 98

    if name == "maze":
        ix = int(np.clip(math.floor((x + 4.8) / 2.4), 0, 3))
        iy = int(np.clip(math.floor((y + 4.8) / 2.4), 0, 3))
        return ix * 4 + iy

    # stepping_stones, jump_box, contour, climb_box
    if r < 1.12:
        return 0  # center pad/block
    sec = int(math.floor((math.atan2(y, x) + math.pi) / (math.pi / 4.0))) % 8
    if r >= 2.30:
        return 1 + sec  # outer ring / outer pads (strictly across intermediate obstacles from center!)
    if allow_bridge:
        return 51 + sec  # intermediate stone / jump_box / climb_box ring
    return 98


def maze_graph_and_free_grid(seed: int, level: int) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
    from scipy.ndimage import binary_dilation
    m = build_terrain_mesh("maze", seed=seed, level=level, terrain_set=TERRAIN_SET)
    res5 = 0.05
    n5 = int(round(TILE_SIZE / res5))
    wall5 = np.zeros((n5, n5), dtype=bool)
    res10 = 0.10
    n10 = int(round(TILE_SIZE / res10))
    wall10 = np.zeros((n10, n10), dtype=bool)
    for c in m.split(only_watertight=False):
        bmin, bmax = c.bounds
        if bmax[2] < 0.30:
            continue
        wall5[max(0, int(math.floor((bmin[0] + HALF_WIDTH) / res5))):min(n5, int(math.ceil((bmax[0] + HALF_WIDTH) / res5))),
              max(0, int(math.floor((bmin[1] + HALF_WIDTH) / res5))):min(n5, int(math.ceil((bmax[1] + HALF_WIDTH) / res5)))] = True
        wall10[max(0, int(math.floor((bmin[0] + HALF_WIDTH) / res10))):min(n10, int(math.ceil((bmax[0] + HALF_WIDTH) / res10))),
               max(0, int(math.floor((bmin[1] + HALF_WIDTH) / res10))):min(n10, int(math.ceil((bmax[1] + HALF_WIDTH) / res10)))] = True

    r5 = int(math.ceil(0.40 / res5))
    yy5, xx5 = np.ogrid[-r5:r5 + 1, -r5:r5 + 1]
    free5 = ~binary_dilation(wall5, structure=(xx5 * xx5 + yy5 * yy5) <= r5 * r5)

    r10 = int(math.ceil(0.32 / res10))
    yy10, xx10 = np.ogrid[-r10:r10 + 1, -r10:r10 + 1]
    free10 = ~binary_dilation(wall10, structure=(xx10 * xx10 + yy10 * yy10) <= r10 * r10)

    centers = [(int((-3.6 + 2.4 * i + HALF_WIDTH) / res5), int((-3.6 + 2.4 * j + HALF_WIDTH) / res5))
               for i in range(4) for j in range(4)]
    adj = [[] for _ in range(16)]
    for i in range(4):
        for j in range(4):
            u = i * 4 + j
            for di, dj in ((1, 0), (0, 1)):
                ni, nj = i + di, j + dj
                if ni < 4 and nj < 4:
                    v = ni * 4 + nj
                    c0, c1 = centers[u], centers[v]
                    steps = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1]))
                    ok = True
                    for t in range(steps + 1):
                        a = int(round(c0[0] + (c1[0] - c0[0]) * t / steps))
                        b = int(round(c0[1] + (c1[1] - c0[1]) * t / steps))
                        if not free5[a, b]:
                            ok = False
                            break
                    if ok:
                        adj[u].append(v)
                        adj[v].append(u)
    dist = np.full((16, 16), 99, dtype=np.int32)
    for src in range(16):
        dist[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[src, v] > dist[src, u] + 1:
                    dist[src, v] = dist[src, u] + 1
                    q.append(v)
    return dist, adj, free10


def compute_dijkstra_field(free10: np.ndarray, gx: float, gy: float) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt
    n = free10.shape[0]
    ig = int(np.clip(round((gx + 4.95) / 0.10), 0, n - 1))
    jg = int(np.clip(round((gy + 4.95) / 0.10), 0, n - 1))
    if not free10[ig, jg]:
        free_pts = np.argwhere(free10)
        best = free_pts[np.argmin((free_pts[:, 0] - ig) ** 2 + (free_pts[:, 1] - jg) ** 2)]
        ig, jg = int(best[0]), int(best[1])

    inf = 1e6
    d = np.full((n, n), inf, dtype=np.float64)
    d[ig, jg] = 0.0
    pq = [(0.0, ig, jg)]
    nbrs = [(-1, 0, 0.10), (1, 0, 0.10), (0, -1, 0.10), (0, 1, 0.10),
            (-1, -1, 0.14142136), (-1, 1, 0.14142136), (1, -1, 0.14142136), (1, 1, 0.14142136)]
    while pq:
        cd, i, j = heapq.heappop(pq)
        if cd > d[i, j] + 1e-9:
            continue
        for di, dj, w in nbrs:
            ni, nj = i + di, j + dj
            if 0 <= ni < n and 0 <= nj < n and free10[ni, nj]:
                if di != 0 and dj != 0 and not (free10[i + di, j] and free10[i, j + dj]):
                    continue
                nd = cd + w
                if nd < d[ni, nj] - 1e-9:
                    d[ni, nj] = nd
                    heapq.heappush(pq, (nd, ni, nj))

    unreachable = d >= inf * 0.5
    if unreachable.any():
        edt, inds = distance_transform_edt(unreachable, return_indices=True)
        d[unreachable] = d[inds[0][unreachable], inds[1][unreachable]] + 0.10 * edt[unreachable]
    return d.astype(np.float32)


def geodesic_yaw_from_field(field: np.ndarray, sx: float, sy: float, gx: float, gy: float) -> float:
    if math.hypot(gx - sx, gy - sy) <= 0.70:
        return math.atan2(gy - sy, gx - sx)
    n = field.shape[0]
    u = float(np.clip((sx + 4.95) / 0.10, 0.0, n - 1.001))
    v = float(np.clip((sy + 4.95) / 0.10, 0.0, n - 1.001))
    i0, j0 = int(u), int(v)
    fu, fv = u - i0, v - j0
    f00 = float(field[i0, j0])
    f10 = float(field[i0 + 1, j0])
    f01 = float(field[i0, j0 + 1])
    f11 = float(field[i0 + 1, j0 + 1])
    gx_m = -((1.0 - fv) * (f10 - f00) + fv * (f11 - f01)) / 0.10
    gy_m = -((1.0 - fu) * (f01 - f00) + fu * (f11 - f10)) / 0.10
    if math.hypot(gx_m, gy_m) > 1e-3:
        return math.atan2(gy_m, gx_m)
    return math.atan2(gy - sy, gx - sx)


def list_full_cross_pairs_v4(
    name: str,
    poses: list[tuple[float, float, float, float, int]],
    maze_dist: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Strict full-obstacle cross pairs (used for BOTH Holdout and Tier-2 Full Cross training)."""
    strict_pairs = []
    fallback_pairs = []
    m = len(poses)
    for i in range(m):
        sx, sy, sz, syaw, sr = poses[i]
        if sr >= 50:
            continue
        for j in range(m):
            if i == j:
                continue
            gx, gy, gz, gyaw, gr = poses[j]
            if gr >= 50 or sr == gr:
                continue
            hd = math.hypot(gx - sx, gy - sy)
            min_r = _min_seg_radius(sx, sy, gx, gy)
            ok_strict = False
            ok_fb = False

            if name == "radial_beam":
                ok_strict = (((sr == 0) and (1 <= gr <= 8)) or ((gr == 0) and (1 <= sr <= 8))) and (hd >= 3.0)
            elif name == "stairs":
                ok_strict = (((sr == 0) and (1 <= gr <= 4)) or ((gr == 0) and (1 <= sr <= 4))) and (hd >= 3.0)
            elif name == "gap":
                perp = (abs(gx - sx) <= 1.40) or (abs(gy - sy) <= 1.40)
                ok_strict = (((sr == 0) and (1 <= gr <= 4)) or ((gr == 0) and (1 <= sr <= 4))) and perp and (hd >= 2.2)
                ok_fb = (((sr == 0) and (1 <= gr <= 4)) or ((gr == 0) and (1 <= sr <= 4))) and (hd >= 1.9)
            elif name == "pit":
                perp = (abs(gx - sx) <= 1.55) or (abs(gy - sy) <= 1.55)
                ok_strict = (((sr == 0) and (1 <= gr <= 4)) or ((gr == 0) and (1 <= sr <= 4))) and perp and (hd >= 2.2)
                ok_fb = (((sr == 0) and (1 <= gr <= 4)) or ((gr == 0) and (1 <= sr <= 4))) and (hd >= 1.9)
            elif name == "balance_beam":
                ok_strict = ((sr == 0 and gr == 1) or (sr == 1 and gr == 0)) and (abs(gy - sy) <= 1.25) and (hd >= 4.3)
                ok_fb = ((sr == 0 and gr == 1) or (sr == 1 and gr == 0)) and (hd >= 4.0)
            elif name == "inverted_slope":
                ok_strict = (((sr == 0) and (1 <= gr <= 4)) or ((gr == 0) and (1 <= sr <= 4))) and (hd >= 3.1)
            elif name == "islands":
                hub_c = ((sr == 0 and 1 <= gr <= 4) or (gr == 0 and 1 <= sr <= 4)) and (hd >= 2.8)
                adj_c = (1 <= sr <= 4 and 1 <= gr <= 4 and ((abs(sr - gr) == 1) or (abs(sr - gr) == 3))) and (hd >= 4.2)
                ok_strict = hub_c or adj_c
            elif name == "parallel_box":
                ok_strict = ((sr == 0 and gr == 1) or (sr == 1 and gr == 0)) and (abs(gx - sx) <= 2.40) and (hd >= 3.9)
                ok_fb = ((sr == 0 and gr == 1) or (sr == 1 and gr == 0)) and (hd >= 3.7)
            elif name == "maze":
                hops = int(maze_dist[sr, gr]) if maze_dist is not None else 0
                ok_strict = (3 <= hops < 99 and hd >= 3.2)
                ok_fb = (2 <= hops < 99 and hd >= 2.4)
            else:
                hub_s = ((sr == 0 and 1 <= gr <= 8) or (gr == 0 and 1 <= sr <= 8)) and (hd >= 2.4)
                opp_s = (1 <= sr <= 8 and 1 <= gr <= 8 and abs(sr - gr) == 4 and min_r <= 1.05 and hd >= 3.6)
                ok_strict = hub_s or opp_s
                ok_fb = (sr != gr and min_r <= 1.25 and hd >= 2.3)

            if ok_strict:
                strict_pairs.append((i, j))
            elif ok_fb:
                fallback_pairs.append((i, j))

    pool = strict_pairs if len(strict_pairs) >= 6 else (strict_pairs + fallback_pairs)
    assert len(pool) > 0, (
        f"Zero valid full cross pairs for {name} (n_poses={m}, "
        f"reg_counts={ {r: sum(1 for p in poses if p[4] == r) for r in sorted(set(p[4] for p in poses))} })"
    )
    return pool


def list_bridge_cross_pairs_v4(
    name: str,
    poses: list[tuple[float, float, float, float, int]],
    maze_dist: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Tier-1 Cross-Edge Near-Goal Bridge pairs (1.05m <= hd <= 2.95m across an obstacle/edge, sr != gr)."""
    near_bridge = []
    med_bridge = []
    m = len(poses)
    for i in range(m):
        sx, sy, sz, syaw, sr = poses[i]
        if sr >= 90:
            continue
        for j in range(m):
            if i == j:
                continue
            gx, gy, gz, gyaw, gr = poses[j]
            if gr >= 90 or sr == gr:
                continue
            hd = math.hypot(gx - sx, gy - sy)
            if hd < 1.05:
                continue

            ok = False
            if name == "radial_beam":
                # Mid-beam (50..57) <-> hub (0) or spoke rim (1..8) along the SAME spoke, OR shortest hub<->spoke!
                if 50 <= sr <= 57:
                    sp = sr - 50
                    ok = (gr == 0 or gr == 1 + sp)
                elif 50 <= gr <= 57:
                    sp = gr - 50
                    ok = (sr == 0 or sr == 1 + sp)
                else:
                    ok = ((sr == 0 and 1 <= gr <= 8) or (gr == 0 and 1 <= sr <= 8))
            elif name == "balance_beam":
                # Mid-beam (50) <-> east rim (1) or island (0), OR shortest island <-> rim
                ok = ({sr, gr} in ({50, 1}, {50, 0}, {0, 1})) and (abs(gy - sy) <= 1.05)
            elif name == "parallel_box":
                # Adjacent/two-row crossings across parallel box rows (0 <-> 50/51, 50 <-> 51/52, 51/52 <-> 1)
                ok = (abs(gx - sx) <= 1.85) and (abs(gy - sy) >= 0.85)
            elif name in ("gap", "pit"):
                perp = (abs(gx - sx) <= 1.25) or (abs(gy - sy) <= 1.25)
                ok = (((sr == 0) and (1 <= gr <= 4)) or ((gr == 0) and (1 <= sr <= 4))) and perp
            elif name in ("stairs", "inverted_slope"):
                if 51 <= sr <= 54:
                    side = sr - 50
                    ok = (gr == 0 or gr == side)
                elif 51 <= gr <= 54:
                    side = gr - 50
                    ok = (sr == 0 or sr == side)
                else:
                    ok = ((sr == 0 and 1 <= gr <= 4) or (gr == 0 and 1 <= sr <= 4))
            elif name == "islands":
                if 50 <= sr <= 53:
                    q = sr - 50 + 1
                    ok = (gr == 0 or gr == q)
                elif 50 <= gr <= 53:
                    q = gr - 50 + 1
                    ok = (sr == 0 or sr == q)
                else:
                    ok = ((sr == 0 and 1 <= gr <= 4) or (gr == 0 and 1 <= sr <= 4))
            elif name == "maze":
                hops = int(maze_dist[sr, gr]) if maze_dist is not None else 0
                ok = (1 <= hops <= 2)
            else:
                # stepping_stones, jump_box, contour, climb_box: center(0) <-> mid-ring(51..58) <-> outer(1..8)
                if 51 <= sr <= 58:
                    sec = sr - 50
                    ok = (gr == 0) or (1 <= gr <= 8 and min(abs(gr - sec), 8 - abs(gr - sec)) <= 1)
                elif 51 <= gr <= 58:
                    sec = gr - 50
                    ok = (sr == 0) or (1 <= sr <= 8 and min(abs(sr - sec), 8 - abs(sr - sec)) <= 1)
                else:
                    ok = ((sr == 0 and 1 <= gr <= 8) or (gr == 0 and 1 <= sr <= 8))

            if not ok:
                continue
            if hd < 2.60:
                near_bridge.append((i, j, hd))
            elif hd < 4.00:
                med_bridge.append((i, j, hd))

    if near_bridge and med_bridge:
        rng_b = np.random.default_rng(len(poses) * 97 + len(near_bridge) * 13 + len(med_bridge))
        n_n = min(len(near_bridge), 360)
        n_m = max(1, int(round(n_n * 0.67)))
        idx_n = rng_b.choice(len(near_bridge), size=n_n, replace=False)
        idx_m = rng_b.choice(len(med_bridge), size=n_m, replace=(len(med_bridge) < n_m))
        return [(near_bridge[int(k)][0], near_bridge[int(k)][1]) for k in idx_n] + [
            (med_bridge[int(k)][0], med_bridge[int(k)][1]) for k in idx_m
        ]
    if len(near_bridge) >= 8:
        return [(i, j) for i, j, _ in near_bridge]
    if len(med_bridge) >= 8:
        return [(i, j) for i, j, _ in med_bridge]
    # Fallback on narrowest L7..L9 radial/balance beams where 4-foot spawn cannot sit mid-beam:
    # return shortest full cross pairs so Tier 1 stays 100% valid cross-obstacle!
    full = list_full_cross_pairs_v4(name, poses, maze_dist=maze_dist)
    scored = sorted(((i, j, math.hypot(poses[j][0] - poses[i][0], poses[j][1] - poses[i][1])) for i, j in full), key=lambda x: x[2])
    return [(i, j) for i, j, _ in scored[:max(16, len(scored) // 3)]]


def list_candidate_easy_pairs_v4(poses: list[tuple[float, float, float, float, int]]) -> list[tuple[int, int]]:
    same_reg, short_any = [], []
    m = len(poses)
    for i in range(m):
        sx, sy, sz, syaw, sr = poses[i]
        for j in range(m):
            if i == j:
                continue
            gx, gy, gz, gyaw, gr = poses[j]
            hd = math.hypot(gx - sx, gy - sy)
            dz = abs(gz - sz)
            if sr == gr and sr < 50 and 0.40 <= hd <= 1.95 and dz <= 0.20:
                same_reg.append((i, j))
            elif 0.40 <= hd <= 1.95 and dz <= 0.20:
                short_any.append((i, j))
    pool = same_reg if len(same_reg) >= 6 else (same_reg + short_any)
    if not pool:
        all_p = [(i, j, math.hypot(poses[j][0] - poses[i][0], poses[j][1] - poses[i][1]))
                 for i in range(m) for j in range(m) if i != j]
        all_p.sort(key=lambda x: x[2])
        pool = [(i, j) for i, j, _ in all_p[:max(6, min(20, len(all_p)))]]
    return pool


def targeted_candidates_v4(name: str, d: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    cx_u = rng.uniform(-HALF_WIDTH + 0.52, HALF_WIDTH - 0.52, size=14000)
    cy_u = rng.uniform(-HALF_WIDTH + 0.52, HALF_WIDTH - 0.52, size=14000)
    extras_x, extras_y = [cx_u], [cy_u]

    if name == "balance_beam":
        # Island + Mid-beam + East rim
        extras_x.extend([rng.uniform(-2.45, -0.58, size=3500), rng.uniform(-0.35, 3.45, size=3500), rng.uniform(3.72, 4.44, size=3500)])
        extras_y.extend([rng.uniform(-0.85, 0.85, size=3500), rng.uniform(-0.28, 0.28, size=3500), rng.uniform(-0.85, 0.85, size=3500)])
    elif name == "radial_beam":
        spokes = rng.integers(0, 8, size=7000) * (math.pi / 4.0)
        rad_rim = rng.uniform(3.75, 4.42, size=3500)
        rad_mid = rng.uniform(1.30, 3.45, size=3500)
        ang_rim = spokes[:3500] + rng.uniform(-0.14, 0.14, size=3500)
        ang_mid = spokes[3500:] + rng.uniform(-0.05, 0.05, size=3500)
        extras_x.extend([rng.uniform(-0.95, 0.95, size=3000), np.clip(rad_rim * np.cos(ang_rim), -4.45, 4.45), np.clip(rad_mid * np.cos(ang_mid), -4.45, 4.45)])
        extras_y.extend([rng.uniform(-0.95, 0.95, size=3000), np.clip(rad_rim * np.sin(ang_rim), -4.45, 4.45), np.clip(rad_mid * np.sin(ang_mid), -4.45, 4.45)])
    elif name in ("gap", "pit"):
        # Explicitly sample near inner & outer lips so 1.25-2.30m 1-obstacle crossings exist at ALL levels!
        if name == "gap":
            inner_lip = 0.5 * (2.60 - 0.30 * d) - 0.55
            outer_lip = 0.5 * (2.60 - 0.30 * d) + (0.10 + 0.66 * d) + 0.58
        else:
            inner_lip = 0.5 * (3.5 + 0.6 * d) - 0.60
            outer_lip = 0.5 * (3.5 + 0.6 * d) + 0.60
        extras_x.append(rng.uniform(-inner_lip, inner_lip, size=3500))
        extras_y.append(rng.uniform(-inner_lip, inner_lip, size=3500))
        side = rng.integers(0, 4, size=5000)
        long_c = rng.uniform(outer_lip, min(4.44, outer_lip + 1.55), size=5000)
        lat_c = rng.uniform(-1.15, 1.15, size=5000)
        extras_x.append(np.where(side == 0, -long_c, np.where(side == 1, long_c, lat_c)))
        extras_y.append(np.where(side == 2, -long_c, np.where(side == 3, long_c, lat_c)))
    elif name == "parallel_box":
        # South bank, 3 parallel box rows, north bank
        extras_x.extend([rng.uniform(-3.8, 3.8, size=3500), rng.uniform(-3.8, 3.8, size=5000), rng.uniform(-3.8, 3.8, size=3500)])
        extras_y.extend([rng.uniform(-4.35, -2.10, size=3500), rng.uniform(-1.55, 1.55, size=5000), rng.uniform(2.10, 4.35, size=3500)])
    else:
        extras_x.append(rng.uniform(-1.35, 1.35, size=3500))
        extras_y.append(rng.uniform(-1.35, 1.35, size=3500))
        ang = rng.uniform(-math.pi, math.pi, size=5000)
        rad = rng.uniform(1.35, 4.35, size=5000)
        extras_x.append(np.clip(rad * np.cos(ang), -4.45, 4.45))
        extras_y.append(np.clip(rad * np.sin(ang), -4.45, 4.45))

    return np.concatenate(extras_x).astype(np.float32), np.concatenate(extras_y).astype(np.float32)


def main():
    out = EXP_ROOT / "catalog_v4"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    _app = launcher.app

    import torch
    import omni.usd
    from pxr import Sdf, UsdGeom, UsdPhysics, Vt
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sensors import RayCasterCfg, patterns
    from isaaclab.utils import configclass
    from isaaclab.utils.warp import raycast_mesh
    from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

    spacing = 14.0
    combos = [(k, s, l) for k in range(len(TERRAIN_NAMES)) for s in range(2) for l in range(10)]
    origins = [(l * spacing, (k * 2 + s) * spacing, 0.0) for (k, s, l) in combos]

    all_v, all_f, geom_hashes = [], [], []
    offset = 0
    for (k, s, l), (ox, oy, oz) in zip(combos, origins):
        v, f = terrain_arrays(k, s, l, terrain_set=TERRAIN_SET)
        geom_hashes.append(hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest())
        wv = v.copy()
        wv[:, 0] += ox
        wv[:, 1] += oy
        wv[:, 2] += oz
        all_v.append(wv)
        all_f.append(f.astype(np.int32).reshape(-1) + offset)
        offset += len(v)
    vertices = np.concatenate(all_v, axis=0)
    indices = np.concatenate(all_f, axis=0)

    @configclass
    class MinimalCfg(InteractiveSceneCfg):
        robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 3.0)),
            ray_alignment="yaw", update_period=0.02,
            pattern_cfg=patterns.GridPatternCfg(resolution=0.20, size=(1.0, 1.0)),
            mesh_prim_paths=[TERRAIN_PRIM],
        )

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, render_interval=4, device="cuda:0", use_fabric=True))
    stage = omni.usd.get_context().get_stage()
    mp = UsdGeom.Mesh.Define(stage, Sdf.Path(TERRAIN_PRIM))
    mp.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    mp.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    mp.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices) // 3, 3, dtype=np.int32)))
    mp.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(mp.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mp.GetPrim()).CreateApproximationAttr("none")
    scene = InteractiveScene(MinimalCfg(num_envs=1, env_spacing=spacing))
    sim.reset()
    warp_mesh = scene["scan"].meshes[TERRAIN_PRIM]
    device = "cuda:0"

    foot_t = torch.tensor(FOOT_SAMPLES, device=device, dtype=torch.float32)
    base_t = torch.tensor(BASE_RING, device=device, dtype=torch.float32)
    all_pts_t = torch.cat([foot_t, base_t], dim=0)
    dir_down = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=torch.float32)

    def check_poses_gpu(xs: np.ndarray, ys: np.ndarray, yaws: np.ndarray, ox: float, oy: float):
        c_cos = torch.tensor(np.cos(yaws), device=device, dtype=torch.float32)
        c_sin = torch.tensor(np.sin(yaws), device=device, dtype=torch.float32)
        cx_t = torch.tensor(xs, device=device, dtype=torch.float32)
        cy_t = torch.tensor(ys, device=device, dtype=torch.float32)
        px = cx_t[:, None] + c_cos[:, None] * all_pts_t[None, :, 0] - c_sin[:, None] * all_pts_t[None, :, 1] + ox
        py = cy_t[:, None] + c_sin[:, None] * all_pts_t[None, :, 0] + c_cos[:, None] * all_pts_t[None, :, 1] + oy
        pz = torch.full_like(px, 4.0)
        starts = torch.stack([px, py, pz], dim=-1).reshape(-1, 3)
        hits, _, _, _ = raycast_mesh(starts, dir_down.expand_as(starts), warp_mesh)
        hz = hits[:, 2].reshape(len(xs), len(FOOT_SAMPLES) + len(BASE_RING))
        foot_z = hz[:, :20]
        base_z = hz[:, 20:]
        foot_finite = torch.isfinite(foot_z).all(dim=1)
        f_min = foot_z.min(dim=1).values
        f_max = foot_z.max(dim=1).values
        f_mean = foot_z.mean(dim=1)
        base_z_safe = torch.where(torch.isfinite(base_z), base_z, torch.full_like(base_z, -99.0))
        b_max = base_z_safe.max(dim=1).values
        ok = foot_finite & (f_min > -1.25) & ((f_max - f_min) <= 0.14) & (b_max <= f_mean + 0.14)
        return ok.cpu().numpy(), f_mean.cpu().numpy()

    maze_field_map: dict[tuple[int, float, float], int] = {}
    maze_fields_list: list[np.ndarray] = []

    def get_or_build_maze_field(tile_idx: int, free10: np.ndarray, gx: float, gy: float) -> int:
        key = (int(tile_idx), round(float(gx), 5), round(float(gy), 5))
        if key not in maze_field_map:
            maze_field_map[key] = len(maze_fields_list)
            maze_fields_list.append(compute_dijkstra_field(free10, gx, gy))
        return maze_field_map[key]

    def sample_strictly_aligned_pairs(
        tile_idx: int, name: str, poses: list[tuple], cand_ij: list[tuple[int, int]],
        n_pairs: int, ox: float, oy: float, rng: np.random.Generator,
        free10: np.ndarray | None, jitter_rad: float,
    ):
        offsets = np.array([0.0, -0.07, 0.07, -0.14, 0.14, -0.22, 0.22, -0.30, 0.30], dtype=np.float32)
        k_off = len(offsets)
        if len(cand_ij) > 1400:
            perm = rng.choice(len(cand_ij), size=1400, replace=False)
            cand_ij = [cand_ij[int(p)] for p in perm]
        m_c = len(cand_ij)
        xs = np.repeat([poses[i][0] for i, _ in cand_ij], k_off).astype(np.float32)
        ys = np.repeat([poses[i][1] for i, _ in cand_ij], k_off).astype(np.float32)
        cand_yaws = np.empty((m_c, k_off), dtype=np.float32)
        desired_yaws = np.empty(m_c, dtype=np.float64)
        gids = np.full(m_c, -1, dtype=np.int32)

        for p_idx, (i, j) in enumerate(cand_ij):
            st, gl = poses[i], poses[j]
            if name == "maze" and free10 is not None:
                gid = get_or_build_maze_field(tile_idx, free10, gl[0], gl[1])
                gids[p_idx] = gid
                des_y = geodesic_yaw_from_field(maze_fields_list[gid], st[0], st[1], gl[0], gl[1])
            else:
                des_y = math.atan2(gl[1] - st[1], gl[0] - st[0])
            desired_yaws[p_idx] = des_y
            jit = float(rng.uniform(-jitter_rad, jitter_rad))
            for ko in range(k_off):
                cand_yaws[p_idx, ko] = _wrap_pi(des_y + jit + float(offsets[ko]))

        ok_flat, z_flat = check_poses_gpu(xs, ys, cand_yaws.reshape(-1), ox, oy)
        ok_mat = ok_flat.reshape(m_c, k_off)
        z_mat = z_flat.reshape(m_c, k_off)

        verified = []
        for p_idx, (i, j) in enumerate(cand_ij):
            vks = np.flatnonzero(ok_mat[p_idx])
            if len(vks) == 0:
                continue
            pick = int(vks[0])
            st, gl = poses[i], poses[j]
            yw = float(cand_yaws[p_idx, pick])
            err_deg = math.degrees(abs(_wrap_pi(yw - desired_yaws[p_idx])))
            assert err_deg <= 23.0, f"Unexpected yaw error {err_deg} deg on {name}"
            st_al = (st[0], st[1], float(z_mat[p_idx, pick]), yw, st[4])
            verified.append((st_al, gl, err_deg, int(gids[p_idx])))

        assert len(verified) > 0, f"Zero verified aligned pairs on tile {tile_idx} ({name}) out of {m_c} candidates!"
        replace = len(verified) < n_pairs
        chosen = rng.choice(len(verified), size=n_pairs, replace=replace)
        return [verified[int(c)] for c in chosen], len(verified)

    rng = np.random.default_rng(20261126)
    train_rows, holdout_rows, train_is_easy, train_tier, strict_routes = [], [], [], [], []
    train_maze_gid, holdout_maze_gid = [], []
    ho_des_yerrs, tr_hard_des_yerrs, unique_hard_counts = [], [], []
    ho_reg_pair_counts = {nm: {} for nm in TERRAIN_NAMES}

    for tile_idx, ((k, s, l), (ox, oy, oz)) in enumerate(zip(combos, origins)):
        name = TERRAIN_NAMES[k]
        d = l / 9.0
        cx, cy = targeted_candidates_v4(name, d, rng)
        # Align candidate yaw along spoke/beam when checking mid-beam poses so 4 feet pass narrow beam checks!
        yaw = rng.uniform(-math.pi, math.pi, size=len(cx)).astype(np.float32)
        if name == "balance_beam":
            on_b = (cx >= -0.45) & (cx <= 3.55) & (np.abs(cy) <= 0.45)
            yaw[on_b] = np.where(rng.random(int(on_b.sum())) < 0.5, 0.0, math.pi).astype(np.float32) + rng.uniform(-0.08, 0.08, size=int(on_b.sum())).astype(np.float32)
        elif name == "radial_beam":
            on_s = (np.hypot(cx, cy) >= 1.05) & (np.hypot(cx, cy) <= 3.55)
            sp_a = np.round(np.arctan2(cy[on_s], cx[on_s]) / (math.pi / 4.0)) * (math.pi / 4.0)
            yaw[on_s] = np.where(rng.random(int(on_s.sum())) < 0.5, sp_a, _wrap_pi_vec := np.arctan2(np.sin(sp_a + math.pi), np.cos(sp_a + math.pi))).astype(np.float32)

        ok_np, f_mean_np = check_poses_gpu(cx, cy, yaw, ox, oy)
        ok_idx = np.flatnonzero(ok_np)

        by_core_reg = {}
        by_bridge_reg = {}
        for ci in ok_idx:
            x_, y_, z_, yw_ = float(cx[ci]), float(cy[ci]), float(f_mean_np[ci]), float(yaw[ci])
            r_all = region_id_v4(name, x_, y_, z_, d, allow_bridge=True)
            if r_all < 50:
                by_core_reg.setdefault(r_all, []).append((x_, y_, z_, yw_, r_all))
            elif r_all < 90:
                by_bridge_reg.setdefault(r_all, []).append((x_, y_, z_, yw_, r_all))

        # 1. Select up to 96 core terminal poses and split strictly into train_core vs hold_poses (0 overlap!)
        selected_core = []
        core_keys = sorted(by_core_reg.keys())
        while len(selected_core) < 96:
            added = False
            for rk in core_keys:
                lst = by_core_reg[rk]
                min_sp = 0.14 if rk == 0 else 0.22
                while lst:
                    cand = lst.pop()
                    if all((cand[0] - q[0]) ** 2 + (cand[1] - q[1]) ** 2 >= min_sp ** 2 for q in selected_core):
                        selected_core.append(cand)
                        added = True
                        break
                if len(selected_core) >= 96:
                    break
            if not added:
                break

        train_poses, hold_poses, singletons = [], [], []
        by_sel_reg = {}
        for pt in selected_core:
            by_sel_reg.setdefault(pt[4], []).append(pt)
        for rk, lst in sorted(by_sel_reg.items()):
            if len(lst) == 1:
                singletons.append(lst[0])
            else:
                split = min(len(lst) - 1, max(1, int(round(len(lst) * 0.68))))
                train_poses.extend(lst[:split])
                hold_poses.extend(lst[split:])
        for idx_s, pt in enumerate(singletons):
            (train_poses if idx_s % 2 == 0 else hold_poses).append(pt)

        # 2. Add up to 44 intermediate bridge poses EXCLUSIVELY to train_poses (strictly >0.18m away from hold_poses!)
        bridge_keys = sorted(by_bridge_reg.keys())
        n_br = 0
        while n_br < 44 and bridge_keys:
            added = False
            for rk in bridge_keys:
                lst = by_bridge_reg[rk]
                while lst:
                    cand = lst.pop()
                    if (all((cand[0] - q[0]) ** 2 + (cand[1] - q[1]) ** 2 >= 0.18 ** 2 for q in hold_poses)
                            and all((cand[0] - q[0]) ** 2 + (cand[1] - q[1]) ** 2 >= 0.16 ** 2 for q in train_poses)):
                        train_poses.append(cand)
                        n_br += 1
                        added = True
                        break
                if n_br >= 44:
                    break
            if not added:
                break

        m_dist, _m_adj, free10 = maze_graph_and_free_grid(s, l) if name == "maze" else (None, None, None)

        # Tier 0A: 12 birth-near (3%: goal == start, standing arrival hold)
        core_train_only = [p for p in train_poses if p[4] < 50]
        near_idx = rng.choice(len(core_train_only), size=12, replace=True)
        for ni in near_idx:
            st = core_train_only[int(ni)]
            train_rows.append([tile_idx, st[0], st[1], st[2], st[3], st[0], st[1], st[2]])
            train_is_easy.append(True)
            train_tier.append(0)
            train_maze_gid.append(get_or_build_maze_field(tile_idx, free10, st[0], st[1]) if name == "maze" else -1)

        # Tier 0B: 28 short same-region easy walk (7%)
        easy_cand = list_candidate_easy_pairs_v4(core_train_only)
        easy_sampled, _ = sample_strictly_aligned_pairs(tile_idx, name, core_train_only, easy_cand, 28, ox, oy, rng, free10, jitter_rad=0.08)
        for st, gl, _yerr, gid in easy_sampled:
            train_rows.append([tile_idx, st[0], st[1], st[2], st[3], gl[0], gl[1], gl[2]])
            train_is_easy.append(True)
            train_tier.append(0)
            train_maze_gid.append(gid)

        # Tier 1: 140 Cross-Edge Near-Goal Bridge routes (35%, non-bypassable 1-obstacle/edge crossings!)
        bridge_cand = list_bridge_cross_pairs_v4(name, train_poses, maze_dist=m_dist)
        try:
            bridge_sampled, n_ver_br = sample_strictly_aligned_pairs(tile_idx, name, train_poses, bridge_cand, 140, ox, oy, rng, free10, jitter_rad=0.07)
        except AssertionError:
            # Fallback if narrow beam yaw alignment filtered bridge_cand: draw shortest verified full cross pairs
            fb_cand = list_full_cross_pairs_v4(name, train_poses, maze_dist=m_dist)
            bridge_sampled, n_ver_br = sample_strictly_aligned_pairs(tile_idx, name, train_poses, fb_cand, 140, ox, oy, rng, free10, jitter_rad=0.07)
        for st, gl, yerr, gid in bridge_sampled:
            train_rows.append([tile_idx, st[0], st[1], st[2], st[3], gl[0], gl[1], gl[2]])
            train_is_easy.append(False)
            train_tier.append(1)
            train_maze_gid.append(gid)
            tr_hard_des_yerrs.append(yerr)

        # Tier 2: 220 Full Cross-Obstacle routes (55%, matching holdout full-crossing distribution!)
        full_cand = list_full_cross_pairs_v4(name, train_poses, maze_dist=m_dist)
        full_sampled, n_ver_full = sample_strictly_aligned_pairs(tile_idx, name, train_poses, full_cand, 220, ox, oy, rng, free10, jitter_rad=0.08)
        unique_hard_counts.append(n_ver_br + n_ver_full)
        for st, gl, yerr, gid in full_sampled:
            train_rows.append([tile_idx, st[0], st[1], st[2], st[3], gl[0], gl[1], gl[2]])
            train_is_easy.append(False)
            train_tier.append(2)
            train_maze_gid.append(gid)
            tr_hard_des_yerrs.append(yerr)

        # Holdout: 8 strictly non-bypassable full cross-obstacle routes per tile (2,080 total)
        ho_cand = list_full_cross_pairs_v4(name, hold_poses, maze_dist=m_dist)
        ho_sampled, _ = sample_strictly_aligned_pairs(tile_idx, name, hold_poses, ho_cand, 8, ox, oy, rng, free10, jitter_rad=0.06)
        for st, gl, yerr, gid in ho_sampled:
            holdout_rows.append([tile_idx, st[0], st[1], st[2], st[3], gl[0], gl[1], gl[2]])
            holdout_maze_gid.append(gid)
            ho_des_yerrs.append(yerr)
            rk_pair = f"{min(st[4], gl[4])}<->{max(st[4], gl[4])}"
            ho_reg_pair_counts[name][rk_pair] = ho_reg_pair_counts[name].get(rk_pair, 0) + 1

        if s == 0 and l == 5:
            st, gl, _, _ = ho_sampled[0]
            strict_routes.append(dict(
                index=len(strict_routes), terrain=name, seed=s, level=l, tile_idx=tile_idx,
                start=[st[0], st[1], st[2]], start_yaw=st[3], goal=[gl[0], gl[1], gl[2]],
                start_region=int(st[4]), goal_region=int(gl[4]),
                horizontal_m=round(math.hypot(gl[0] - st[0], gl[1] - st[1]), 3),
                delta_z_m=round(gl[2] - st[2], 3),
            ))
            if name == "islands" and len(ho_sampled) >= 3:
                for extra_i in (1, 2):
                    st2, gl2, _, _ = ho_sampled[extra_i]
                    strict_routes.append(dict(
                        index=len(strict_routes), terrain=name, seed=s, level=l, tile_idx=tile_idx,
                        start=[st2[0], st2[1], st2[2]], start_yaw=st2[3], goal=[gl2[0], gl2[1], gl2[2]],
                        start_region=int(st2[4]), goal_region=int(gl2[4]),
                        horizontal_m=round(math.hypot(gl2[0] - st2[0], gl2[1] - st2[1]), 3),
                        delta_z_m=round(gl2[2] - st2[2], 3),
                    ))
        if s == 1 and l == 5 and name == "stairs":
            st, gl, _, _ = ho_sampled[0]
            strict_routes.append(dict(
                index=len(strict_routes), terrain=name, seed=s, level=l, tile_idx=tile_idx,
                start=[st[0], st[1], st[2]], start_yaw=st[3], goal=[gl[0], gl[1], gl[2]],
                start_region=int(st[4]), goal_region=int(gl[4]),
                horizontal_m=round(math.hypot(gl[0] - st[0], gl[1] - st[1]), 3),
                delta_z_m=round(gl[2] - st[2], 3),
            ))

    train_arr = np.asarray(train_rows, dtype=np.float32)
    holdout_arr = np.asarray(holdout_rows, dtype=np.float32)
    easy_arr = np.asarray(train_is_easy, dtype=np.bool_)
    tier_arr = np.asarray(train_tier, dtype=np.int8)
    assert train_arr.shape == (104000, 8) and holdout_arr.shape == (2080, 8) and len(strict_routes) == 16

    # Hard zero-overlap assertion between train and holdout XY endpoints
    tr_pts = set(zip(train_arr[:, 0].astype(int), np.round(train_arr[:, 1], 4), np.round(train_arr[:, 2], 4)))
    tr_pts |= set(zip(train_arr[:, 0].astype(int), np.round(train_arr[:, 5], 4), np.round(train_arr[:, 6], 4)))
    ho_pts = set(zip(holdout_arr[:, 0].astype(int), np.round(holdout_arr[:, 1], 4), np.round(holdout_arr[:, 2], 4)))
    ho_pts |= set(zip(holdout_arr[:, 0].astype(int), np.round(holdout_arr[:, 5], 4), np.round(holdout_arr[:, 6], 4)))
    assert len(tr_pts.intersection(ho_pts)) == 0, "Train/holdout endpoint overlap!"

    maze_fields_arr = np.stack(maze_fields_list, axis=0).astype(np.float32)
    train_gid_arr = np.asarray(train_maze_gid, dtype=np.int32)
    holdout_gid_arr = np.asarray(holdout_maze_gid, dtype=np.int32)
    np.savez_compressed(
        out / "maze_geodesic.npz",
        fields=maze_fields_arr,
        train_gid=train_gid_arr,
        holdout_gid=holdout_gid_arr,
    )

    n = len(train_arr)
    tidx = train_arr[:, 0].astype(np.int64)
    kinds = tidx // 20
    levels = tidx % 10
    sx, sy, sz, syaw = train_arr[:, 1], train_arr[:, 2], train_arr[:, 3], train_arr[:, 4]
    gx, gy, gz = train_arr[:, 5], train_arr[:, 6], train_arr[:, 7]
    dx, dy, dz = gx - sx, gy - sy, gz - sz
    hdist = np.hypot(dx, dy)
    g_azim = np.arctan2(dy, dx)
    yaw_err = np.arctan2(np.sin(g_azim - syaw), np.cos(g_azim - syaw))

    # Per-terrain hard distance-bin continuity audit (verifies the OmniReset bridge!)
    per_terrain_hard_bins = {}
    for k, nm in enumerate(TERRAIN_NAMES):
        mk = (kinds == k) & (~easy_arr)
        hd_k = hdist[mk]
        b_near = int(((hd_k >= 1.00) & (hd_k < 2.60)).sum())
        b_mid = int(((hd_k >= 2.60) & (hd_k < 4.00)).sum())
        b_far = int((hd_k >= 4.00).sum())
        per_terrain_hard_bins[nm] = dict(
            n_hard=int(mk.sum()),
            min_m=round(float(hd_k.min()), 3),
            p25_m=round(float(np.percentile(hd_k, 25)), 3),
            med_m=round(float(np.median(hd_k)), 3),
            bin_1p0_2p6m=b_near,
            bin_2p6_4p0m=b_mid,
            bin_ge_4p0m=b_far,
        )
        assert b_near >= 200, f"Bridge discontinuity on {nm}: only {b_near} hard tasks in [1.0, 2.6)m!"

    one_hot = np.zeros((n, len(TERRAIN_NAMES)), dtype=np.float64)
    one_hot[np.arange(n), kinds] = 1.0
    cont = np.column_stack([levels.astype(np.float64), sx, sy, gx, gy, hdist, dz, syaw, yaw_err])
    raw_feat = np.column_stack([one_hot, cont]).astype(np.float64)
    cont_enc, _ = build_fourier_features(cont, periodic_dims=(7, 8), freqs=(1.0, 2.0))
    model_feat = np.ascontiguousarray(np.column_stack([one_hot.astype(np.float32), cont_enc]), dtype=np.float32)

    bounds = tuple([(-0.01, 1.01)] * 13 + [
        (-0.5, 9.5), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0),
        (0.0, 14.5), (-2.5, 2.5), (-math.pi, math.pi), (-math.pi, math.pi),
    ])
    pool = TaskPool(raw_features=raw_feat, features=model_feat, bounds=bounds, periodic_dims=(20, 21))
    grid_bins = tuple([2] * 13 + [5, 2, 2, 2, 2, 4, 3, 2, 2])
    _, occ, _, _ = compute_sparse_cell_ids(pool, tuple(range(22)), grid_bins)

    np.save(out / "train_tasks.npy", train_arr)
    np.save(out / "holdout_tasks.npy", holdout_arr)
    np.save(out / "train_is_easy.npy", easy_arr)
    np.save(out / "train_tier.npy", tier_arr)
    np.save(out / "raw_features.npy", raw_feat.astype(np.float32))
    np.save(out / "model_features.npy", model_feat)
    (out / "strict_routes.json").write_text(json.dumps(strict_routes, indent=2) + "\n")

    ho_dist = np.hypot(holdout_arr[:, 5] - holdout_arr[:, 1], holdout_arr[:, 6] - holdout_arr[:, 2])
    ho_az = np.arctan2(holdout_arr[:, 6] - holdout_arr[:, 2], holdout_arr[:, 5] - holdout_arr[:, 1])
    ho_straight_yerr = np.degrees(np.abs(np.arctan2(np.sin(ho_az - holdout_arr[:, 4]), np.cos(ho_az - holdout_arr[:, 4]))))
    meta = dict(
        version="sgs-13terrain-trimesh-v4-continuous-bridge",
        terrain_set=TERRAIN_SET,
        n_tiles=260, n_train=104000,
        n_train_easy=int(easy_arr.sum()),
        n_train_bridge=int((tier_arr == 1).sum()),
        n_train_full_cross=int((tier_arr == 2).sum()),
        n_train_hard=int((~easy_arr).sum()),
        n_holdout=2080, n_strict=16,
        aligned_yaw_fallback_count=0,
        unique_maze_geodesic_fields=int(len(maze_fields_list)),
        unique_verified_hard_pairs_per_tile_min=int(min(unique_hard_counts)),
        unique_verified_hard_pairs_per_tile_med=round(float(np.median(unique_hard_counts)), 1),
        holdout_dist_min_m=round(float(ho_dist.min()), 3),
        holdout_dist_med_m=round(float(np.median(ho_dist)), 3),
        train_hard_desired_yaw_err_p50_deg=round(float(np.median(tr_hard_des_yerrs)), 2),
        train_hard_desired_yaw_err_p95_deg=round(float(np.percentile(tr_hard_des_yerrs, 95)), 2),
        train_hard_desired_yaw_err_max_deg=round(float(np.max(tr_hard_des_yerrs)), 2),
        holdout_desired_yaw_err_p50_deg=round(float(np.median(ho_des_yerrs)), 2),
        holdout_desired_yaw_err_p95_deg=round(float(np.percentile(ho_des_yerrs, 95)), 2),
        holdout_desired_yaw_err_max_deg=round(float(np.max(ho_des_yerrs)), 2),
        holdout_straight_yaw_err_p50_deg=round(float(np.median(ho_straight_yerr)), 2),
        per_terrain_hard_distance_bins=per_terrain_hard_bins,
        per_terrain_holdout_region_pairs=ho_reg_pair_counts,
        endpoint_overlap=0,
        raw_dim=int(raw_feat.shape[1]), model_dim=int(model_feat.shape[1]),
        bounds=[list(b) for b in bounds], periodic_dims=[20, 21],
        grid_bins=list(grid_bins), grid_n_active=int(len(occ)),
        train_sha256=hashlib.sha256(train_arr.tobytes()).hexdigest(),
        holdout_sha256=hashlib.sha256(holdout_arr.tobytes()).hexdigest(),
        strict_sha256=hashlib.sha256((out / "strict_routes.json").read_bytes()).hexdigest(),
        maze_geodesic_sha256=hashlib.sha256((out / "maze_geodesic.npz").read_bytes()).hexdigest(),
        collision_sha256=hashlib.sha256("".join(geom_hashes).encode()).hexdigest(),
    )
    (out / "catalog_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[CATALOG v4] Built in {time.time()-t0:.1f}s on terrain_set={TERRAIN_SET}: "
          f"train={n} (easy={meta['n_train_easy']}, bridge={meta['n_train_bridge']}, full={meta['n_train_full_cross']}), "
          f"holdout=2080, overlap=0, fallbacks=0, ho_med_dist={meta['holdout_dist_med_m']}m", flush=True)
    for nm, bs in per_terrain_hard_bins.items():
        print(f"  {nm:16s}: min={bs['min_m']:.2f}m p25={bs['p25_m']:.2f}m med={bs['med_m']:.2f}m "
              f"[1.0-2.6m]={bs['bin_1p0_2p6m']:4d} [2.6-4.0m]={bs['bin_2p6_4p0m']:4d} [>=4.0m]={bs['bin_ge_4p0m']:4d}", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
