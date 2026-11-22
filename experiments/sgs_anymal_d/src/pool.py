"""The fixed SGS-style task dictionary and independently seeded evaluation draws."""

from argparse import ArgumentParser
from hashlib import sha256
import json
from math import dist, hypot
from pathlib import Path
import random
from struct import pack

from navigation import BASE_HEIGHT, controlled_cases
from terrain import (GEOMETRY_VERSION, LEVELS, PATCHES, TASK_COUNT, TERRAIN_NAMES, TERRAIN_SEEDS,
                     birth_lift, decode, encode, height_at, mesh, tile)


TRAIN_STREAM = 247040
EVAL_STREAM = 432173


def task(task_id):
    kind, seed, level, start, goal = decode(task_id)
    geometry = tile(kind, seed, level)
    return geometry.starts[start], geometry.goals[goal]


def stream(seed, count):
    rng = random.Random(seed)
    return [rng.randrange(TASK_COUNT) for _ in range(count)]


def audit():
    pool = sha256()
    initial = sha256()
    coverage = {}
    per_terrain_hashes = {}
    minimum_distance = float("inf")
    max_triangles = 0
    for kind, name in enumerate(TERRAIN_NAMES):
        terrain_coordinates = sha256()
        tiles = []
        for seed in range(TERRAIN_SEEDS):
            for level in range(LEVELS):
                geometry = tile(kind, seed, level)
                patches = geometry.starts + geometry.goals
                assert len(set(patches)) == 2 * PATCHES
                assert all(height_at(geometry.grid, x, y) == z for x, y, z in patches)
                vertices, faces = mesh(geometry.grid)
                max_triangles = max(max_triangles, len(faces))
                assert min(i for face in faces for i in face) == 0
                assert max(i for face in faces for i in face) < len(vertices)
                assert all(len(set(face)) == 3 for face in faces)
                for start, a in enumerate(geometry.starts):
                    for goal, b in enumerate(geometry.goals):
                        task_id = encode(kind, seed, level, start, goal)
                        assert decode(task_id) == (kind, seed, level, start, goal)
                        minimum_distance = min(minimum_distance, hypot(a[0]-b[0], a[1]-b[1]))
                        packed = pack("<I6d", task_id, *a, *b)
                        pool.update(packed)
                        initial.update(pack("<I3d",task_id,a[0],a[1],a[2]+BASE_HEIGHT+birth_lift(task_id)))
                        terrain_coordinates.update(packed)
                tiles.append(dict(seed=seed, level=level, vertices=len(vertices), triangles=len(faces),
                                  id_first=encode(kind, seed, level, 0, 0),
                                  id_last=encode(kind, seed, level, PATCHES-1, PATCHES-1)))
        coverage[name] = tiles
        per_terrain_hashes[name] = terrain_coordinates.hexdigest()
    assert minimum_distance >= .64 and TASK_COUNT == 104_000
    training_before = stream(TRAIN_STREAM, 32)
    evaluation = stream(EVAL_STREAM, 128)
    assert training_before == stream(TRAIN_STREAM, 32) and evaluation == stream(EVAL_STREAM, 128)
    assert training_before != evaluation[:32]
    return dict(geometry_version=GEOMETRY_VERSION, total_tasks=TASK_COUNT, terrain_seed_count=TERRAIN_SEEDS, levels=LEVELS,
                starts_per_tile=PATCHES, goals_per_tile=PATCHES, task_id_location_sha256=pool.hexdigest(),
                task_id_initial_root_sha256=initial.hexdigest(),
                max_tile_triangles=max_triangles, minimum_start_goal_xy_distance=minimum_distance,
                coverage=coverage, terrain_task_locations_sha256=per_terrain_hashes,
                training_seed=TRAIN_STREAM, evaluation_seed=EVAL_STREAM,
                example_training_ids=training_before, evaluation_ids=evaluation,
                coordinate_examples=[dict(task_id=i, start=task(i)[0], goal=task(i)[1],
                                          distance=dist(*task(i))) for i in training_before[:6]],
                controlled=controlled_cases())


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(), indent=2, allow_nan=False) + "\n")
