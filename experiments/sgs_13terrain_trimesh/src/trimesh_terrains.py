"""1:1 Trimesh 3D reconstructions of the 13 SGS locomotion terrains.

All terrains are pure watertight/solid trimesh surfaces in a 10m x 10m tile
centered at (0, 0) (i.e. [-5, +5] x [-5, +5]), with nominal top elevation z=0
and abyss/drop threshold at z=-1.8m.

Drop-in compatible with Isaac Lab UsdGeom.Mesh + UsdPhysics.MeshCollisionAPI
+ Warp RayCaster and offline Metal rendering.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Tuple

import numpy as np
import trimesh

TERRAIN_NAMES = (
    "gap",
    "pit",
    "radial_beam",
    "stairs",
    "parallel_box",
    "balance_beam",
    "stepping_stones",
    "jump_box",
    "maze",
    "contour",
    "islands",
    "climb_box",
    "inverted_slope",
)

TILE_SIZE = 10.0
HALF_WIDTH = TILE_SIZE / 2.0
FLOOR_Z = -2.2


def _box(sx: float, sy: float, sz: float, cx: float, cy: float, cz: float, yaw: float = 0.0) -> trimesh.Trimesh:
    m = trimesh.creation.box((float(sx), float(sy), float(sz)))
    if yaw != 0.0:
        m.apply_transform(trimesh.transformations.rotation_matrix(float(yaw), [0.0, 0.0, 1.0]))
    m.apply_translation((float(cx), float(cy), float(cz)))
    return m


def _box_top(sx: float, sy: float, top_z: float, cx: float, cy: float, yaw: float = 0.0, bottom_z: float = FLOOR_Z) -> trimesh.Trimesh:
    sz = max(0.05, float(top_z) - float(bottom_z))
    cz = float(bottom_z) + 0.5 * sz
    return _box(sx, sy, sz, cx, cy, cz, yaw=yaw)


def _box_thick(sx: float, sy: float, thickness: float, top_z: float, cx: float, cy: float, yaw: float = 0.0) -> trimesh.Trimesh:
    return _box(sx, sy, thickness, cx, cy, float(top_z) - 0.5 * float(thickness), yaw=yaw)


def _prism_thick(radius: float, sections: int, thickness: float, top_z: float, cx: float, cy: float, yaw: float = 0.0) -> trimesh.Trimesh:
    m = trimesh.creation.cylinder(radius=float(radius), height=float(thickness), sections=int(sections))
    if yaw != 0.0:
        m.apply_transform(trimesh.transformations.rotation_matrix(float(yaw), [0.0, 0.0, 1.0]))
    m.apply_translation((float(cx), float(cy), float(top_z) - 0.5 * float(thickness)))
    return m


def _border(outer: float, inner_x: float, inner_y: float, top_z: float = 0.0, bottom_z: float = FLOOR_Z, cx: float = 0.0, cy: float = 0.0) -> list[trimesh.Trimesh]:
    tx = (outer - inner_x) / 2.0
    ty = (outer - inner_y) / 2.0
    meshes = []
    if ty > 1e-4:
        meshes.append(_box_top(outer, ty, top_z, cx, cy + inner_y / 2.0 + ty / 2.0, bottom_z=bottom_z))
        meshes.append(_box_top(outer, ty, top_z, cx, cy - inner_y / 2.0 - ty / 2.0, bottom_z=bottom_z))
    if tx > 1e-4:
        meshes.append(_box_top(tx, inner_y, top_z, cx - inner_x / 2.0 - tx / 2.0, cy, bottom_z=bottom_z))
        meshes.append(_box_top(tx, inner_y, top_z, cx + inner_x / 2.0 + tx / 2.0, cy, bottom_z=bottom_z))
    return meshes


def _boulder(rng: np.random.Generator, radius: float, cx: float, cy: float, cz: float, height_scale: float = 1.15) -> trimesh.Trimesh:
    """Create a crisp faceted low-poly rock boulder via perturbed convex hull."""
    pts = rng.normal(size=(24, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True).clip(1e-6)
    scales = rng.uniform(0.65, 1.22, size=(len(pts), 1))
    pts *= scales * radius
    pts[:, 2] *= height_scale
    hull = trimesh.convex.convex_hull(pts)
    hull.apply_translation((float(cx), float(cy), float(cz)))
    return hull


def _build_gap(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Central square island surrounded by a square moat inside outer flat ground."""
    platform_w = 2.6 - 0.35 * d
    gap_w = 0.22 + 0.65 * d
    inner = platform_w + 2.0 * gap_w
    island_dz = float(rng.uniform(-0.06, 0.12) * d)
    meshes = []
    meshes.extend(_border(TILE_SIZE, inner, inner, top_z=0.0, bottom_z=-1.6))
    meshes.append(_box_thick(platform_w, platform_w, 1.6, island_dz, 0.0, 0.0))
    return meshes


def _build_pit(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Outer flat ground with a crisp square recessed pit in the center."""
    depth = 0.16 + 0.62 * d
    pit_w = 3.5 + 0.6 * d
    meshes = []
    meshes.extend(_border(TILE_SIZE, pit_w, pit_w, top_z=0.0, bottom_z=-1.6))
    meshes.append(_box_top(pit_w, pit_w, -depth, 0.0, 0.0, bottom_z=-1.6))
    return meshes


def _build_radial_beam(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Outer border around void + central octagonal hub + 8 floating radial beams."""
    inner = 7.4
    hub_r = 1.15 - 0.15 * d
    beam_w = 0.60 - 0.32 * d
    beam_thick = 0.45
    meshes = []
    meshes.extend(_border(TILE_SIZE, inner, inner, top_z=0.0, bottom_z=-1.6))
    meshes.append(_prism_thick(hub_r, 8, beam_thick, 0.0, 0.0, 0.0, yaw=math.pi / 8.0))
    for k in range(8):
        ang = k * (math.pi / 4.0)
        reach = (inner * 0.52) / max(abs(math.cos(ang)), abs(math.sin(ang)))
        length = reach - hub_r * 0.65
        mid = hub_r * 0.65 + 0.5 * length
        cx, cy = mid * math.cos(ang), mid * math.sin(ang)
        meshes.append(_box_thick(length, beam_w, beam_thick, 0.0, cx, cy, yaw=ang))
    return meshes


def _build_stairs(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """4-sided pyramidal staircase rising from outer border to central platform."""
    border = 1.15
    step_w = 0.28
    step_h = 0.055 + 0.145 * d
    platform_w = 2.0
    n_steps = int((TILE_SIZE - 2.0 * border - platform_w) // (2.0 * step_w))
    meshes = [_box_thick(TILE_SIZE, TILE_SIZE, 0.6, 0.0, 0.0, 0.0)]
    for i in range(1, n_steps + 1):
        span = TILE_SIZE - 2.0 * border - 2.0 * (i - 1) * step_w
        if span <= platform_w:
            break
        z_top = i * step_h
        meshes.append(_box_top(span, span, z_top, 0.0, 0.0, bottom_z=(i - 1) * step_h - 0.04))
    z_peak = (n_steps + 1) * step_h
    meshes.append(_box_top(platform_w, platform_w, z_peak, 0.0, 0.0, bottom_z=n_steps * step_h - 0.04))
    return meshes


def _build_parallel_box(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Two parallel clusters of yaw-rotated cubic blocks separated by a stepping chasm."""
    meshes = []
    # Matches SGS Par. Box: crisp ~1.0m cubes floating over dark void in 2 parallel sides + bridging stones
    for side in (-1.0, 1.0):
        # Cluster along diagonal/parallel ridge
        for i in range(-3, 4):
            for row in (0, 1):
                cx = i * 1.18 + rng.uniform(-0.18, 0.18)
                cy = side * (2.25 + row * 1.15) + rng.uniform(-0.22, 0.22)
                if abs(cx) > HALF_WIDTH - 0.6 or abs(cy) > HALF_WIDTH - 0.6:
                    continue
                if row == 1 and rng.random() < 0.25:
                    continue
                sx = rng.uniform(1.05, 1.38) - 0.08 * d
                sy = rng.uniform(1.05, 1.38) - 0.08 * d
                yaw = rng.uniform(-0.55, 0.55)
                z_top = rng.uniform(-0.15 * d, 0.35 * d)
                meshes.append(_box_thick(sx, sy, 1.05, z_top, cx, cy, yaw=yaw))
    # 3 bridging stepping cubes across the middle void between the two parallel ridges
    bridge_x = [-2.2, 0.0, 2.2]
    for bx in bridge_x:
        for by in (-0.85, 0.85):
            if rng.random() < 0.35 and abs(bx) > 0.5:
                continue
            cx = bx + rng.uniform(-0.20, 0.20)
            cy = by * (0.85 + 0.25 * d) + rng.uniform(-0.12, 0.12)
            sx = rng.uniform(0.90, 1.12) - 0.10 * d
            sy = rng.uniform(0.90, 1.12) - 0.10 * d
            z_top = rng.uniform(-0.10 * d, 0.25 * d)
            meshes.append(_box_thick(sx, sy, 1.05, z_top, cx, cy, yaw=rng.uniform(-0.45, 0.45)))
    return meshes


def _build_balance_beam(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Outer square rim + offset square island connected by a floating slotted/ribbed beam."""
    inner = 7.4
    meshes = []
    meshes.extend(_border(TILE_SIZE, inner, inner, top_z=0.0, bottom_z=-1.6))
    plat_w = 2.2
    plat_x = -1.65
    meshes.append(_box_thick(plat_w, plat_w, 1.2, 0.0, plat_x, 0.0))
    beam_start = plat_x + 0.5 * plat_w
    beam_end = 0.5 * inner
    span = beam_end - beam_start
    beam_w = 0.72 - 0.36 * d
    if d < 0.15:
        meshes.append(_box_thick(span + 0.1, beam_w, 0.45, 0.0, 0.5 * (beam_start + beam_end), 0.0))
    else:
        n_rungs = 8
        pitch = span / n_rungs
        gap = min(pitch * 0.48, 0.04 + 0.15 * d)
        rung_len = pitch - gap
        for i in range(n_rungs):
            rx = beam_start + (i + 0.5) * pitch
            meshes.append(_box_thick(rung_len, beam_w, 0.65, 0.0, rx, 0.0))
    return meshes


def _build_stepping_stones(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Central flat square pad + 4 outer quadrant landing pads inside dense stepping-stone matrix."""
    meshes = []
    plat_w = 1.85
    meshes.append(_box_thick(plat_w, plat_w, 1.0, 0.0, 0.0, 0.0))
    pad_w = 1.40
    pads = [(-3.65, -3.65), (3.65, -3.65), (3.65, 3.65), (-3.65, 3.65)]
    for (px, py) in pads:
        pz = float(rng.uniform(-0.04 * d, 0.06 * d))
        meshes.append(_box_thick(pad_w, pad_w, 1.0, pz, px, py))
    pitch = 0.76
    gap = 0.08 + 0.17 * d
    stone_w = pitch - gap
    half_n = int((HALF_WIDTH - 0.35) // pitch)
    for i in range(-half_n, half_n + 1):
        for j in range(-half_n, half_n + 1):
            cx = i * pitch + rng.uniform(-0.025, 0.025) * d
            cy = j * pitch + rng.uniform(-0.025, 0.025) * d
            if abs(cx) < 0.5 * plat_w + 0.5 * stone_w - 0.05 and abs(cy) < 0.5 * plat_w + 0.5 * stone_w - 0.05:
                continue
            if any(abs(cx - px) < 0.5 * pad_w + 0.5 * stone_w - 0.05 and abs(cy - py) < 0.5 * pad_w + 0.5 * stone_w - 0.05 for px, py in pads):
                continue
            z_top = float(rng.uniform(-0.03 - 0.09 * d, 0.03 + 0.09 * d))
            meshes.append(_box_thick(stone_w, stone_w, 1.0, z_top, cx, cy))
    return meshes


def _build_jump_box(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Hexagonal pillars mixed with square/rectangular blocks floating over void."""
    meshes = []
    gap = 0.12 + 0.40 * d
    # Central cluster of 3 square blocks (like SGS Jump Box center) + crisp hexagonal columns
    meshes.append(_box_thick(1.55, 1.55, 1.05, 0.0, 0.0, 0.0))
    for (ux, uy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sx = rng.uniform(1.30, 1.55) - 0.12 * d
        sy = rng.uniform(1.30, 1.55) - 0.12 * d
        cx = ux * (0.775 + 0.5 * sx + gap) + rng.uniform(-0.08, 0.08)
        cy = uy * (0.775 + 0.5 * sy + gap) + rng.uniform(-0.08, 0.08)
        z_top = float(rng.uniform(-0.10 * d, 0.30 * d))
        meshes.append(_box_thick(sx, sy, 1.05, z_top, cx, cy, yaw=rng.uniform(-0.15, 0.15)))
    hex_r = 0.90 - 0.10 * d
    ring_coords = [
        (-2.35, -2.15), (-0.70, -3.40), (1.40, -2.90), (2.75, -1.60),
        (2.85, 1.50), (1.45, 3.00), (-1.25, 2.80), (-2.75, 1.50),
        (-3.55, -0.35), (3.65, -0.15),
    ]
    for (bx, by) in ring_coords:
        scale = 0.90 + 0.14 * d
        cx = bx * scale + rng.uniform(-0.16, 0.16)
        cy = by * scale + rng.uniform(-0.16, 0.16)
        if abs(cx) > HALF_WIDTH - hex_r - 0.1 or abs(cy) > HALF_WIDTH - hex_r - 0.1:
            continue
        z_top = float(rng.uniform(-0.15 * d, 0.34 * d))
        meshes.append(_prism_thick(hex_r, 6, 1.05, z_top, cx, cy, yaw=rng.uniform(0.0, math.pi / 3.0)))
    return meshes


def _build_maze(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Guaranteed-solvable 4x4 cell spanning-tree labyrinth with crisp thin concrete walls."""
    wall_h = 1.35
    wall_t = 0.20
    meshes = [_box_thick(TILE_SIZE, TILE_SIZE, 0.5, 0.0, 0.0, 0.0)]
    meshes.extend(_border(TILE_SIZE, TILE_SIZE - 2 * wall_t, TILE_SIZE - 2 * wall_t, top_z=wall_h, bottom_z=-0.05))

    # Build a 4x4 cell grid spanning [-4.8, +4.8] -> cell size = 2.4m (plenty of clearance for ANYmal-D!).
    # Generate a guaranteed connected spanning tree via randomized DFS, then open a couple extra
    # doors at low difficulty so every cell is 100% reachable from every other cell!
    nx, ny = 4, 4
    inner_span = TILE_SIZE - 2.0 * wall_t  # 9.6m
    cell = inner_span / nx                 # 2.4m
    x0 = -0.5 * inner_span
    y0 = -0.5 * inner_span

    visited = [[False] * ny for _ in range(nx)]
    open_edges = set()
    stack = [(int(rng.integers(0, nx)), int(rng.integers(0, ny)))]
    visited[stack[0][0]][stack[0][1]] = True
    while stack:
        cx, cy = stack[-1]
        nbrs = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx_, ny_ = cx + dx, cy + dy
            if 0 <= nx_ < nx and 0 <= ny_ < ny and not visited[nx_][ny_]:
                nbrs.append((nx_, ny_))
        if nbrs:
            nx_, ny_ = nbrs[int(rng.integers(0, len(nbrs)))]
            visited[nx_][ny_] = True
            edge = tuple(sorted([(cx, cy), (nx_, ny_)]))
            open_edges.add(edge)
            stack.append((nx_, ny_))
        else:
            stack.pop()

    # Open extra loops at easier levels (0 extra at hardest d=1.0 -> pure winding tree maze)
    all_edges = []
    for i in range(nx):
        for j in range(ny):
            if i + 1 < nx:
                all_edges.append(((i, j), (i + 1, j)))
            if j + 1 < ny:
                all_edges.append(((i, j), (i, j + 1)))
    closed = [e for e in all_edges if e not in open_edges]
    extra = int(round((1.0 - d) * 4))
    if extra > 0 and closed:
        for idx in rng.choice(len(closed), size=min(extra, len(closed)), replace=False):
            open_edges.add(closed[int(idx)])

    # For open passages, optionally add partial doorway jambs at high difficulty (keeping doorway >= 1.45m wide!)
    jamb_frac = 0.22 * d  # doorway stays >= (1 - 2*0.22)*2.4 = 1.34m wide even at L9!
    for i in range(nx):
        for j in range(ny):
            # Vertical wall between (i, j) and (i+1, j)
            if i + 1 < nx:
                wx = x0 + (i + 1) * cell
                yc = y0 + (j + 0.5) * cell
                if ((i, j), (i + 1, j)) not in open_edges:
                    meshes.append(_box_top(wall_t, cell + wall_t, wall_h, wx, yc, bottom_z=-0.05))
                elif jamb_frac > 0.04:
                    jl = cell * jamb_frac
                    meshes.append(_box_top(wall_t, jl, wall_h, wx, y0 + j * cell + 0.5 * jl, bottom_z=-0.05))
                    meshes.append(_box_top(wall_t, jl, wall_h, wx, y0 + (j + 1) * cell - 0.5 * jl, bottom_z=-0.05))
            # Horizontal wall between (i, j) and (i, j+1)
            if j + 1 < ny:
                wy = y0 + (j + 1) * cell
                xc = x0 + (i + 0.5) * cell
                if ((i, j), (i, j + 1)) not in open_edges:
                    meshes.append(_box_top(cell + wall_t, wall_t, wall_h, xc, wy, bottom_z=-0.05))
                elif jamb_frac > 0.04:
                    jl = cell * jamb_frac
                    meshes.append(_box_top(jl, wall_t, wall_h, x0 + i * cell + 0.5 * jl, wy, bottom_z=-0.05))
                    meshes.append(_box_top(jl, wall_t, wall_h, x0 + (i + 1) * cell - 0.5 * jl, wy, bottom_z=-0.05))
    return meshes


def _build_contour(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """High-res stepped topographic contour terraces + faceted 3D boulders & ridges."""
    n = 128
    xs = np.linspace(-HALF_WIDTH, HALF_WIDTH, n)
    ys = np.linspace(-HALF_WIDTH, HALF_WIDTH, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    phi = rng.uniform(0.0, 2.0 * math.pi, size=6)
    h = (
        0.55 * np.sin(0.55 * X + 0.32 * Y + phi[0])
        + 0.42 * np.cos(0.48 * Y - 0.40 * X + phi[1])
        + 0.24 * np.sin(1.05 * X - 0.80 * Y + phi[2])
    )
    r_center = np.hypot(X, Y)
    center_blend = np.clip((r_center - 0.90) / 1.10, 0.0, 1.0)
    amp = 0.28 + 0.60 * d
    z_raw = h * amp * center_blend
    # Bold stepped topographic strata (clean wide ledges + crisp cliff risers)
    step = 0.16 + 0.06 * d
    phase = z_raw / step
    k_floor = np.floor(phase)
    frac = phase - k_floor
    smooth_step = np.clip((frac - 0.78) / 0.22, 0.0, 1.0)
    Z = (k_floor + smooth_step) * step

    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    idx = np.arange(n * n).reshape(n, n)
    f1 = np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()])
    f2 = np.column_stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()])
    faces = np.vstack([f1, f2])
    ground = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    meshes = [ground]

    # Add 6-8 bold faceted rocks + fractured blocks protruding from terraces
    n_rocks = 6 + int(round(3 * d))
    for i in range(n_rocks):
        ang = (i / n_rocks) * 2.0 * math.pi + rng.uniform(-0.3, 0.3)
        dist = rng.uniform(2.0, 4.1)
        rx, ry = dist * math.cos(ang), dist * math.sin(ang)
        ix = int(np.clip((rx + HALF_WIDTH) / TILE_SIZE * (n - 1), 0, n - 1))
        iy = int(np.clip((ry + HALF_WIDTH) / TILE_SIZE * (n - 1), 0, n - 1))
        rz = float(Z[ix, iy])
        rad = rng.uniform(0.55, 1.10 + 0.35 * d)
        meshes.append(_boulder(rng, rad, rx, ry, rz + 0.22 * rad, height_scale=rng.uniform(0.9, 1.4)))
    return meshes


def _build_islands(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """5 floating octagonal islands linked by stepping-stone bridges over dark void."""
    meshes = []
    center_r = 1.45 - 0.15 * d
    corner_r = 1.15 - 0.12 * d
    center_z = 0.12 + 0.36 * d
    corner_pos = [(-3.25, -3.25), (3.25, -3.25), (3.25, 3.25), (-3.25, 3.25)]
    corner_z = [float(rng.uniform(-0.08 * d, 0.08 * d)) for _ in range(4)]

    island_thick = 0.75
    stone_thick = 0.55
    meshes.append(_prism_thick(center_r, 8, island_thick, center_z, 0.0, 0.0, yaw=math.pi / 8.0))
    for (cx, cy), cz in zip(corner_pos, corner_z):
        meshes.append(_prism_thick(corner_r, 8, island_thick, cz, cx, cy, yaw=math.pi / 8.0))

    def add_bridge(p0, r0, z0, p1, r1, z1, n_stones: int):
        v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float64)
        dist = float(np.linalg.norm(v))
        u = v / dist
        ang = math.atan2(u[1], u[0])
        s0 = r0 * 0.90
        s1 = dist - r1 * 0.90
        span = s1 - s0
        if span <= 0.2:
            return
        pitch = span / n_stones
        gap = min(pitch * 0.45, 0.06 + 0.14 * d)
        stone_len = pitch - gap
        stone_w = 0.58 - 0.18 * d
        for k in range(n_stones):
            t = (k + 0.5) / n_stones
            pos = np.array(p0) + u * (s0 + t * span)
            z = (1.0 - t) * z0 + t * z1 + rng.uniform(-0.03, 0.03) * d
            meshes.append(_box_thick(stone_len, stone_w, stone_thick, z, float(pos[0]), float(pos[1]), yaw=ang))

    for (cx, cy), cz in zip(corner_pos, corner_z):
        add_bridge((0.0, 0.0), center_r, center_z, (cx, cy), corner_r, cz, n_stones=5)
    for i in range(4):
        j = (i + 1) % 4
        add_bridge(corner_pos[i], corner_r, corner_z[i], corner_pos[j], corner_r, corner_z[j], n_stones=7)
    return meshes


def _build_climb_box(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Dense urban parkour field of randomly yaw-rotated climbable cubic boxes over void."""
    meshes = []
    # Matches SGS Climb box: dense overlapping staggered cubes (0.0 -> 0.95m) with open void corners
    max_h = 0.28 + 0.72 * d
    # Central landing box
    meshes.append(_box_thick(1.5, 1.5, 1.1, 0.0, 0.0, 0.0))
    for gx in np.linspace(-3.6, 3.6, 6):
        for gy in np.linspace(-3.6, 3.6, 6):
            # Leave one corner open abyss like the SGS Climb box thumbnail
            if gx < -1.8 and gy > 1.8:
                continue
            cx = float(gx + rng.uniform(-0.32, 0.32))
            cy = float(gy + rng.uniform(-0.32, 0.32))
            if math.hypot(cx, cy) < 0.85:
                continue
            if rng.random() < 0.12:
                continue
            sx = float(rng.uniform(0.95, 1.48))
            sy = float(rng.uniform(0.95, 1.48))
            yaw = float(rng.uniform(0.0, math.pi))
            tier = rng.choice([0.05, 0.35, 0.65, 1.0], p=[0.28, 0.30, 0.24, 0.18])
            z_top = float(tier * max_h + rng.uniform(-0.05, 0.06))
            meshes.append(_box_thick(sx, sy, 1.25 + max(0.0, z_top), z_top, cx, cy, yaw=yaw))
    # One faceted boulder perched near the open corner like SGS Climb box
    meshes.append(_boulder(rng, 0.85 + 0.20 * d, -2.5, 3.2, 0.35, height_scale=1.25))
    return meshes


def _build_inverted_slope(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Crisp 4-sided smooth inverted pyramid slope descending to a recessed central platform."""
    border = 1.2
    plat_w = 2.0
    depth = 0.22 + 0.68 * d
    outer_r = HALF_WIDTH - border
    inner_r = 0.5 * plat_w
    meshes = []
    meshes.extend(_border(TILE_SIZE, 2.0 * outer_r, 2.0 * outer_r, top_z=0.0, bottom_z=-1.6))
    meshes.append(_box_top(plat_w, plat_w, -depth, 0.0, 0.0, bottom_z=-1.6))
    o, i = outer_r, inner_r
    outer_pts = [(-o, -o, 0.0), (o, -o, 0.0), (o, o, 0.0), (-o, o, 0.0)]
    inner_pts = [(-i, -i, -depth), (i, -i, -depth), (i, i, -depth), (-i, i, -depth)]
    verts = np.array(outer_pts + inner_pts, dtype=np.float64)
    faces = []
    for k in range(4):
        kn = (k + 1) % 4
        faces.append([k, kn, 4 + kn])
        faces.append([k, 4 + kn, 4 + k])
    meshes.append(trimesh.Trimesh(vertices=verts, faces=np.array(faces, dtype=np.int64), process=False))
    return meshes


BUILDERS = {
    "gap": _build_gap,
    "pit": _build_pit,
    "radial_beam": _build_radial_beam,
    "stairs": _build_stairs,
    "parallel_box": _build_parallel_box,
    "balance_beam": _build_balance_beam,
    "stepping_stones": _build_stepping_stones,
    "jump_box": _build_jump_box,
    "maze": _build_maze,
    "contour": _build_contour,
    "islands": _build_islands,
    "climb_box": _build_climb_box,
    "inverted_slope": _build_inverted_slope,
}


# ---------------------------------------------------------------------------
# V6 Continuous-Curriculum Builders (terrain_set="v6"):
# Eliminates L0 physical dead-ends (e.g. beam_w < 0.636m foot span, slotted beams,
# 1.0m vertical walls, 4.5m parallel-box void) so L0 is 100% walkable by basic
# locomotion while L5..L9 expands smoothly into full dynamic parkour over abyss.
# ---------------------------------------------------------------------------

def _build_gap_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    platform_w = 2.60 - 0.30 * d
    gap_w = 0.10 + 0.66 * d  # L0=0.10m (walkable) -> L5=0.47m -> L9=0.76m leap
    inner = platform_w + 2.0 * gap_w
    island_dz = float(rng.uniform(-0.04, 0.09) * d)
    meshes = []
    meshes.extend(_border(TILE_SIZE, inner, inner, top_z=0.0, bottom_z=-1.6))
    meshes.append(_box_thick(platform_w, platform_w, 1.6, island_dz, 0.0, 0.0))
    return meshes


def _build_pit_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    depth = 0.10 + 0.62 * d  # L0=0.10m -> L5=0.44m -> L9=0.72m
    pit_w = 3.5 + 0.6 * d
    meshes = []
    meshes.extend(_border(TILE_SIZE, pit_w, pit_w, top_z=0.0, bottom_z=-1.6))
    meshes.append(_box_top(pit_w, pit_w, -depth, 0.0, 0.0, bottom_z=-1.6))
    return meshes


def _build_radial_beam_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    inner = 7.4
    hub_r = 1.35 - 0.15 * d
    beam_w = 0.92 - 0.52 * d  # L0=0.92m (> 0.636m foot span!) -> L5=0.63m -> L9=0.40m cat-walk
    beam_thick = 0.45
    meshes = []
    meshes.extend(_border(TILE_SIZE, inner, inner, top_z=0.0, bottom_z=-1.6))
    meshes.append(_prism_thick(hub_r, 8, beam_thick, 0.0, 0.0, 0.0, yaw=math.pi / 8.0))
    for k in range(8):
        ang = k * (math.pi / 4.0)
        reach = (inner * 0.52) / max(abs(math.cos(ang)), abs(math.sin(ang)))
        length = reach - hub_r * 0.60
        mid = hub_r * 0.60 + 0.5 * length
        cx, cy = mid * math.cos(ang), mid * math.sin(ang)
        meshes.append(_box_thick(length, beam_w, beam_thick, 0.0, cx, cy, yaw=ang))
    return meshes


def _build_parallel_box_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Two solid outer landing banks (|y| >= 1.85m) + 3 parallel staggered rows of 3D boxes over void."""
    meshes = []
    bank_depth = HALF_WIDTH - 1.85  # 3.15m deep banks on south (y < -1.85) and north (y > +1.85)
    meshes.append(_box_thick(TILE_SIZE - 0.4, bank_depth, 1.15, 0.0, 0.0, -1.85 - 0.5 * bank_depth))
    meshes.append(_box_thick(TILE_SIZE - 0.4, bank_depth, 1.15, 0.0, 0.0, +1.85 + 0.5 * bank_depth))

    # 3 parallel rows of boxes spanning y in [-1.85, +1.85] (total span = 3.70m -> 4 gaps + 3 rows)
    gap_y = 0.05 + 0.25 * d          # L0=0.05m -> L5=0.19m -> L9=0.30m void gap between parallel rows
    row_sy = (3.70 - 4.0 * gap_y) / 3.0  # L0=1.167m -> L9=0.833m wide boxes
    row_centers_y = [
        -1.85 + gap_y + 0.5 * row_sy,
        0.0,
        +1.85 - gap_y - 0.5 * row_sy,
    ]
    n_cols = 7
    pitch_x = 8.40 / n_cols          # 1.20m pitch along x
    gap_x = 0.04 + 0.22 * d          # L0=0.04m -> L9=0.26m gap along x
    for r_idx, ry in enumerate(row_centers_y):
        stagger = (0.28 * d) * (1.0 if r_idx == 1 else -0.5)
        for c_idx in range(n_cols):
            cx = -4.20 + (c_idx + 0.5) * pitch_x + stagger + float(rng.uniform(-0.08, 0.08) * d)
            cx = float(np.clip(cx, -HALF_WIDTH + 0.75, HALF_WIDTH - 0.75))
            cy = ry + float(rng.uniform(-0.05, 0.05) * d)
            sx = pitch_x - gap_x + float(rng.uniform(-0.04, 0.04) * d)
            sy = row_sy + float(rng.uniform(-0.04, 0.04) * d)
            z_top = float(rng.uniform(-0.12 * d, 0.24 * d))
            yaw = float(rng.uniform(-0.20 * d, 0.20 * d))
            meshes.append(_box_thick(sx, sy, 1.15, z_top, cx, cy, yaw=yaw))
    return meshes


def _build_balance_beam_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Outer square rim + offset floating island connected by a continuous solid narrowing beam."""
    inner = 7.2
    meshes = []
    meshes.extend(_border(TILE_SIZE, inner, inner, top_z=0.0, bottom_z=-1.6))
    plat_w = 2.30
    plat_x = -1.55  # [-2.70, -0.40]: 0.90m west abyss gap prevents jumping west to border!
    island_z = float(rng.uniform(-0.04, 0.08) * d)
    meshes.append(_box_thick(plat_w, plat_w, 1.2, island_z, plat_x, 0.0))
    beam_start = plat_x + 0.5 * plat_w - 0.06
    beam_end = 0.5 * inner + 0.06
    span = beam_end - beam_start
    beam_w = 0.92 - 0.52 * d  # L0=0.92m -> L5=0.63m -> L9=0.40m solid balance beam
    beam_z = 0.5 * island_z
    meshes.append(_box_thick(span, beam_w, 0.55, beam_z, 0.5 * (beam_start + beam_end), 0.0))
    return meshes


def _build_stepping_stones_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    meshes = []
    plat_w = 2.00
    meshes.append(_box_thick(plat_w, plat_w, 1.0, 0.0, 0.0, 0.0))
    pad_w = 1.60
    pads = [(-3.55, -3.55), (3.55, -3.55), (3.55, 3.55), (-3.55, 3.55)]
    for (px, py) in pads:
        pz = float(rng.uniform(-0.04 * d, 0.06 * d))
        meshes.append(_box_thick(pad_w, pad_w, 1.0, pz, px, py))
    pitch = 0.72
    gap = 0.04 + 0.21 * d  # L0=0.04m hairline -> L5=0.16m -> L9=0.25m stone gap over void
    stone_w = pitch - gap
    half_n = int((HALF_WIDTH - 0.30) // pitch)
    for i in range(-half_n, half_n + 1):
        for j in range(-half_n, half_n + 1):
            cx = i * pitch + float(rng.uniform(-0.025, 0.025) * d)
            cy = j * pitch + float(rng.uniform(-0.025, 0.025) * d)
            if abs(cx) < 0.5 * plat_w + 0.5 * stone_w - 0.06 and abs(cy) < 0.5 * plat_w + 0.5 * stone_w - 0.06:
                continue
            if any(abs(cx - px) < 0.5 * pad_w + 0.5 * stone_w - 0.06 and abs(cy - py) < 0.5 * pad_w + 0.5 * stone_w - 0.06 for px, py in pads):
                continue
            z_top = float(rng.uniform(-0.10 * d, 0.11 * d))
            meshes.append(_box_thick(stone_w, stone_w, 1.0, z_top, cx, cy))
    return meshes


def _build_jump_box_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Structured 5x5 parkour grid of square blocks and hexagonal pillars over abyss."""
    meshes = []
    pitch = 1.80
    gap = 0.05 + 0.35 * d  # L0=0.05m seam -> L5=0.24m -> L9=0.40m jump chasm!
    block_w = pitch - gap
    for ix in range(-2, 3):
        for iy in range(-2, 3):
            if ix == 0 and iy == 0:
                meshes.append(_box_thick(max(block_w, 1.65), max(block_w, 1.65), 1.15, 0.0, 0.0, 0.0))
                continue
            cx = ix * pitch + float(rng.uniform(-0.07, 0.07) * d)
            cy = iy * pitch + float(rng.uniform(-0.07, 0.07) * d)
            ring = max(abs(ix), abs(iy))
            z_top = float(rng.uniform(-0.08 * d, (0.18 if ring == 1 else 0.28) * d))
            if (abs(ix) + abs(iy)) % 2 == 1 or ring == 1:
                sx = block_w + float(rng.uniform(-0.04, 0.04) * d)
                sy = block_w + float(rng.uniform(-0.04, 0.04) * d)
                meshes.append(_box_thick(sx, sy, 1.15, z_top, cx, cy, yaw=float(rng.uniform(-0.16 * d, 0.16 * d))))
            else:
                # Hexagonal outer pillar (inradius = r*cos(30deg) = 0.866*r -> match block_w/2)
                hex_r = (0.5 * block_w) / 0.8660254
                meshes.append(_prism_thick(hex_r, 6, 1.15, z_top, cx, cy, yaw=float(rng.uniform(0.0, math.pi / 3.0))))
    return meshes


def _build_contour_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    n = 128
    xs = np.linspace(-HALF_WIDTH, HALF_WIDTH, n)
    ys = np.linspace(-HALF_WIDTH, HALF_WIDTH, n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    phi = rng.uniform(0.0, 2.0 * math.pi, size=6)
    h = (
        0.55 * np.sin(0.55 * X + 0.32 * Y + phi[0])
        + 0.42 * np.cos(0.48 * Y - 0.40 * X + phi[1])
        + 0.24 * np.sin(1.05 * X - 0.80 * Y + phi[2])
    )
    r_center = np.hypot(X, Y)
    center_blend = np.clip((r_center - 0.95) / 1.10, 0.0, 1.0)
    amp = 0.14 + 0.68 * d
    z_raw = h * amp * center_blend
    step = 0.10 + 0.11 * d  # L0=0.10m risers -> L9=0.21m cliff terraces
    phase = z_raw / step
    k_floor = np.floor(phase)
    frac = phase - k_floor
    smooth_step = np.clip((frac - 0.78) / 0.22, 0.0, 1.0)
    Z = (k_floor + smooth_step) * step

    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float64)
    idx = np.arange(n * n).reshape(n, n)
    f1 = np.column_stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()])
    f2 = np.column_stack([idx[:-1, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()])
    faces = np.vstack([f1, f2])
    ground = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    meshes = [ground]

    n_rocks = int(round(2 + 6 * d))
    for i in range(n_rocks):
        ang = (i / max(1, n_rocks)) * 2.0 * math.pi + float(rng.uniform(-0.3, 0.3))
        dist = float(rng.uniform(2.2, 4.1))
        rx, ry = dist * math.cos(ang), dist * math.sin(ang)
        ix = int(np.clip((rx + HALF_WIDTH) / TILE_SIZE * (n - 1), 0, n - 1))
        iy = int(np.clip((ry + HALF_WIDTH) / TILE_SIZE * (n - 1), 0, n - 1))
        rz = float(Z[ix, iy])
        rad = float(rng.uniform(0.45, 0.85 + 0.45 * d))
        meshes.append(_boulder(rng, rad, rx, ry, rz + 0.20 * rad, height_scale=float(rng.uniform(0.85, 1.3))))
    return meshes


def _build_islands_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    meshes = []
    center_r = 1.55 - 0.15 * d
    corner_r = 1.25 - 0.10 * d
    center_z = 0.04 + 0.36 * d
    corner_pos = [(-3.25, -3.25), (3.25, -3.25), (3.25, 3.25), (-3.25, 3.25)]
    corner_z = [float(rng.uniform(-0.07 * d, 0.07 * d)) for _ in range(4)]

    island_thick = 0.75
    stone_thick = 0.55
    meshes.append(_prism_thick(center_r, 8, island_thick, center_z, 0.0, 0.0, yaw=math.pi / 8.0))
    for (cx, cy), cz in zip(corner_pos, corner_z):
        meshes.append(_prism_thick(corner_r, 8, island_thick, cz, cx, cy, yaw=math.pi / 8.0))

    def add_bridge(p0, r0, z0, p1, r1, z1, n_stones: int):
        v = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=np.float64)
        dist = float(np.linalg.norm(v))
        u = v / dist
        ang = math.atan2(u[1], u[0])
        s0 = r0 * 0.86
        s1 = dist - r1 * 0.86
        span = s1 - s0
        if span <= 0.2:
            return
        pitch = span / n_stones
        gap = min(pitch * 0.44, 0.03 + 0.17 * d)  # L0=0.03m -> L9=0.20m
        stone_len = pitch - gap
        stone_w = 0.90 - 0.44 * d                  # L0=0.90m (>0.636m!) -> L5=0.66m -> L9=0.46m
        for k in range(n_stones):
            t = (k + 0.5) / n_stones
            pos = np.array(p0) + u * (s0 + t * span)
            z = (1.0 - t) * z0 + t * z1 + float(rng.uniform(-0.025, 0.025) * d)
            meshes.append(_box_thick(stone_len, stone_w, stone_thick, z, float(pos[0]), float(pos[1]), yaw=ang))

    for (cx, cy), cz in zip(corner_pos, corner_z):
        add_bridge((0.0, 0.0), center_r, center_z, (cx, cy), corner_r, cz, n_stones=5)
    for i in range(4):
        j = (i + 1) % 4
        add_bridge(corner_pos[i], corner_r, corner_z[i], corner_pos[j], corner_r, corner_z[j], n_stones=7)
    return meshes


def _build_climb_box_v6(rng: np.random.Generator, d: float) -> list[trimesh.Trimesh]:
    """Dense 7x7 parkour grid of climbable boxes (L0 max_h=0.08m -> L9 max_h=0.52m, max single step <= 0.28m)."""
    meshes = []
    max_h = 0.08 + 0.44 * d  # L0=0.08m -> L5=0.32m -> L9=0.52m (physically climbable by ANYmal-D!)
    # Smooth radial/random tier map on 7x7 grid so adjacent boxes differ by <= 1.2 tiers (no 1.0m vertical walls!)
    n_g = 7
    coords = np.linspace(-3.84, 3.84, n_g)
    raw_tier = rng.uniform(0.15, 1.0, size=(n_g, n_g))
    # Smooth once with neighbors to avoid impossible isolated sheer towers while keeping crisp multi-tier steps
    smooth = 0.55 * raw_tier + 0.45 * np.pad(raw_tier, 1, mode="edge")[1:-1, 1:-1]
    smooth[3, 3] = 0.0  # central landing box at z=0.0
    meshes.append(_box_thick(1.65, 1.65, 1.2, 0.0, 0.0, 0.0))
    for i, gx in enumerate(coords):
        for j, gy in enumerate(coords):
            if i == 3 and j == 3:
                continue
            # Preserve open corner accent outside main sectors at high difficulty only
            if i == 0 and j == n_g - 1 and d > 0.3:
                continue
            cx = float(gx + rng.uniform(-0.14, 0.14) * d)
            cy = float(gy + rng.uniform(-0.14, 0.14) * d)
            sx = float(rng.uniform(1.30, 1.48))
            sy = float(rng.uniform(1.30, 1.48))
            yaw = float(rng.uniform(-0.35 * d, 0.35 * d))
            q_tier = round(float(smooth[i, j]) * 3.0) / 3.0
            z_top = float(q_tier * max_h + rng.uniform(-0.02, 0.03) * d)
            meshes.append(_box_thick(sx, sy, 1.25 + max(0.0, z_top), z_top, cx, cy, yaw=yaw))
    return meshes


BUILDERS_V6 = {
    "gap": _build_gap_v6,
    "pit": _build_pit_v6,
    "radial_beam": _build_radial_beam_v6,
    "stairs": _build_stairs,
    "parallel_box": _build_parallel_box_v6,
    "balance_beam": _build_balance_beam_v6,
    "stepping_stones": _build_stepping_stones_v6,
    "jump_box": _build_jump_box_v6,
    "maze": _build_maze,
    "contour": _build_contour_v6,
    "islands": _build_islands_v6,
    "climb_box": _build_climb_box_v6,
    "inverted_slope": _build_inverted_slope,
}


@lru_cache(maxsize=520)
def build_terrain_mesh(kind: int | str, seed: int = 0, level: int = 5, *, terrain_set: str = "v3") -> trimesh.Trimesh:
    name = TERRAIN_NAMES[kind] if isinstance(kind, int) else str(kind)
    k = TERRAIN_NAMES.index(name)
    d = float(np.clip(level / 9.0, 0.0, 1.0))
    rng = np.random.default_rng([20261123, k, int(seed), int(level)])
    table = BUILDERS_V6 if terrain_set == "v6" else BUILDERS
    if terrain_set not in ("v3", "v6"):
        raise ValueError(f"Unsupported terrain_set: {terrain_set}")
    parts = table[name](rng, d)
    return trimesh.util.concatenate(parts)


def terrain_arrays(kind: int | str, seed: int = 0, level: int = 5, *, terrain_set: str = "v3") -> Tuple[np.ndarray, np.ndarray]:
    m = build_terrain_mesh(kind, seed, level, terrain_set=terrain_set)
    return np.asarray(m.vertices, dtype="<f4"), np.asarray(m.faces, dtype="<u4")


if __name__ == "__main__":
    import json
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "preview_meshes"
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for idx, name in enumerate(TERRAIN_NAMES):
        m = build_terrain_mesh(name, seed=0, level=5)
        v = np.asarray(m.vertices, dtype="<f4")
        f = np.asarray(m.faces, dtype="<u4")
        fv = v[f].reshape(-1, 3).astype("<f4")
        fname = f"{idx:02d}-{name}.f32"
        (out / fname).write_bytes(fv.tobytes())
        empty = f"{idx:02d}-{name}_floor.f32"
        (out / empty).write_bytes(b"")
        manifest.append(dict(index=idx, name=name, file=fname, main_tris=int(len(fv)//3), floor_file=empty, floor_tris=0))
        print(f"{idx:02d} {name:16s} verts={len(v):6d} faces={len(f):6d} z=[{v[:,2].min():+.2f}, {v[:,2].max():+.2f}]")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
