"""Export locally supplied USD visual meshes relative to their simulated body frames."""
import json
from pathlib import Path
import numpy as np


def export_visuals(world, folder):
    from pxr import Usd, UsdGeom
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    cache = UsdGeom.XformCache()

    def save_geometry(name, points, faces, directory):
        points = np.ascontiguousarray(points, dtype='<f4')
        faces = np.ascontiguousarray(faces, dtype='<u4')
        (directory / (name + '.mesh')).write_bytes(points.tobytes() + faces.tobytes())
        return dict(file=name + '.mesh', vertices=len(points), triangles=len(faces))

    parts = []
    for body, name in enumerate(world.robot.body_names):
        body_path = f'/World/envs/env_0/Robot/{name}'
        prim = world.stage.GetPrimAtPath(body_path + '/visuals')
        if not prim:
            continue
        for part in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if not part.IsA(UsdGeom.Mesh):
                continue
            visual = UsdGeom.Mesh(part)
            local, _ = cache.ComputeRelativeTransform(part, world.stage.GetPrimAtPath(body_path))
            matrix = np.asarray(local, dtype=np.float32)
            points = np.asarray(visual.GetPointsAttr().Get(), dtype=np.float32)
            points = points @ matrix[:3, :3] + matrix[3, :3]
            indices = np.asarray(visual.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            counts = np.asarray(visual.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
            if np.all(counts == 3):
                faces = indices.reshape(-1, 3)
            else:
                faces = []
                offset = 0
                for count in counts:
                    face = indices[offset:offset + count]
                    faces.extend((face[0], face[i], face[i + 1]) for i in range(1, count - 1))
                    offset += count
            if len(faces):
                parts.append(dict(save_geometry(f'part{len(parts):02d}', points, faces, folder), body=body, name=name))
    if len({part['body'] for part in parts}) < 13:
        raise RuntimeError('Official visual mesh incomplete')
    (folder/'robot.json').write_text(json.dumps(parts,indent=2))

