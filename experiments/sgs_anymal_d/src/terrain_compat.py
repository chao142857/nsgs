"""Shared terrain indexes and height-grid triangulation."""
TERRAIN_NAMES = (
    "gap", "pit", "radial_beam", "stairs", "parallel_box", "balance_beam",
    "stepping_stones", "jump_box", "maze", "contour", "islands", "climb_box",
    "inverted_slope",
)
BASE_GEOMETRY_VERSION = "sgs-anymald-local-v2-parallel-box-footprint"
TERRAIN_SEEDS = 2
LEVELS = 10
PATCHES = 20
GRID = 80
RESOLUTION = 0.15
HALF_WIDTH = GRID * RESOLUTION / 2
DROP = -1.8
TILE_COUNT = len(TERRAIN_NAMES) * TERRAIN_SEEDS * LEVELS
TASK_COUNT = TILE_COUNT * PATCHES * PATCHES
def encode(kind, seed, level, start, goal):
    values = (kind, seed, level, start, goal)
    bounds = (len(TERRAIN_NAMES), TERRAIN_SEEDS, LEVELS, PATCHES, PATCHES)
    if any(not 0 <= val < bound for val, bound in zip(values, bounds)):
        raise ValueError(f"Out-of-range task coordinates {values}")
    return ((((kind * TERRAIN_SEEDS + seed) * LEVELS + level) * PATCHES + start) * PATCHES + goal)


def decode(task_id):
    if not 0 <= task_id < TASK_COUNT:
        raise ValueError(f"Out-of-range task ID {task_id}")
    tile, goal = divmod(task_id, PATCHES)
    tile, start = divmod(tile, PATCHES)
    terrain, level = divmod(tile, LEVELS)
    kind, seed = divmod(terrain, TERRAIN_SEEDS)
    return kind, seed, level, start, goal


def mesh(grid):
    """Create an upward-oriented collision surface with actual vertical risers."""
    vertices, faces, indices = [], [], {}

    def quad(corners):
        indices_face = []
        for corner in corners:
            if corner not in indices:
                indices[corner] = len(vertices)
                vertices.append(corner)
            indices_face.append(indices[corner])
        a, b, c, d = indices_face
        faces.extend(((a, b, c), (a, c, d)))

    bounds = [i * RESOLUTION - HALF_WIDTH for i in range(GRID+1)]
    for i in range(GRID):
        for j in range(GRID):
            x, nx = bounds[i:i+2]
            y, ny = bounds[j:j+2]
            z = grid[i][j]
            quad(((x,y,z),(nx,y,z),(nx,ny,z),(x,ny,z)))
            if i+1 < GRID and (adjacent := grid[i+1][j]) != z:
                quad(((nx,y,z),(nx,y,adjacent),(nx,ny,adjacent),(nx,ny,z)))
            if j+1 < GRID and (adjacent := grid[i][j+1]) != z:
                quad(((x,ny,z),(nx,ny,z),(nx,ny,adjacent),(x,ny,adjacent)))
    return vertices, faces
