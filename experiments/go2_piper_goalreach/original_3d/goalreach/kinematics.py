"""Position-only native USD-chain audit. No simulator imports or orientation goal."""
import json
from pathlib import Path
import numpy as np


class Arm:
    def __init__(self, path):
        self.data = json.loads(Path(path).read_text())
        joints = {j['name']: j for j in self.data['joints']}
        self.chain = [joints[f'joint{i}'] for i in range(1, 7)] + [joints['ee_gripper']]
        self.lo = np.array([j['lower'] for j in self.chain[:6]]) + 1e-4
        self.hi = np.array([j['upper'] for j in self.chain[:6]]) - 1e-4
        self.t0 = np.array([j['T0'] for j in self.chain])
        self.t1 = np.linalg.inv(np.array([j['T1'] for j in self.chain]))
        m = joints['connector_to_arm_joint']
        self.mount = np.array(m['T0']) @ np.linalg.inv(m['T1'])
        # Triangle-inequality bound for fixed base, all possible joint rotations.
        self.radial_bound = np.linalg.norm(self.mount[:2, 3]) + sum(
            np.linalg.norm(t[:3, 3]) for t in [self.t0[0]] +
            [self.t1[i] @ self.t0[i+1] for i in range(6)] + [self.t1[-1]])
        self.nominal = self.fk(np.zeros((1, 6)))[0]

    def fk(self, q, return_links=False):
        q = np.atleast_2d(q)
        t = np.broadcast_to(self.mount, (len(q), 4, 4)).copy()
        t[:, 2, 3] += .35
        points = [t[:, :3, 3].copy()]
        for i, j in enumerate(self.chain):
            r = np.broadcast_to(np.eye(4), t.shape).copy()
            if i < 6:
                a = 'XYZ'.index(j['axis'])
                b, c = (a+1)%3, (a+2)%3
                cs, sn = np.cos(q[:, i]), np.sin(q[:, i])
                r[:, b, b] = r[:, c, c] = cs
                r[:, b, c], r[:, c, b] = -sn, sn
            t = t @ self.t0[i] @ r @ self.t1[i]
            points.append(t[:, :3, 3].copy())
        return np.stack(points, 1) if return_links else t[:, :3, 3]

    def admissible(self, q):
        links = self.fk(q, True)
        xyz = links[:, -1]
        # Exclude below-ground links and inflated chassis volume. This conservative
        # segment test is NOT a full mesh/self-collision or dynamic feasibility proof.
        samples = np.concatenate([links[:, 2:-1]*(1-t)+links[:, 3:]*t
                                  for t in np.linspace(0, 1, 9)], axis=1)
        chassis = ((np.abs(samples[..., 0]) < .32) & (np.abs(samples[..., 1]) < .20)
                   & (samples[..., 2] < .48))
        return ((xyz[:, 2] >= .10) & (xyz[:, 2] <= .85)
                & (samples[..., 2] >= .07).all(1) & ~chassis.any(1))

