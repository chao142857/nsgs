"""Read authored collision bounds and transform them with actual simulated body poses."""
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
import torch


class CollisionAudit:
    def __init__(self,env):
        from pxr import Usd,UsdGeom,UsdPhysics
        from isaaclab.utils.math import quat_apply
        self.quat_apply=quat_apply
        usd=Path(env.cfg.scene.robot.spawn.usd_path)
        stage=Usd.Stage.Open(str(usd))
        # Collision geometry is authored invisible; visibility must not discard it.
        cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
        transforms=UsdGeom.XformCache(Usd.TimeCode.Default())
        self.entries=[]
        for i,name in enumerate(env.scene['robot'].body_names):
            body=stage.GetPrimAtPath('/go2_piper/'+name)
            if not body:continue
            for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or prim.GetAttribute('physics:collisionEnabled').Get() is False:continue
                bound=cache.ComputeRelativeBound(prim,body).ComputeAlignedRange()
                lo,hi=np.array(bound.GetMin()),np.array(bound.GetMax())
                assert np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all(),f'Invalid collision bound {prim.GetPath()} {prim.GetTypeName()} {lo} {hi}'
                corners=np.array(list(itertools.product(*zip(lo,hi))),dtype=np.float32)
                entry=dict(body_id=i,body=name,path=str(prim.GetPath()),corners=corners)
                if prim.IsA(UsdGeom.Sphere):
                    relative=np.asarray(transforms.GetLocalToWorldTransform(prim)*transforms.GetLocalToWorldTransform(body).GetInverse())
                    entry['sphere_center']=relative[3,:3].tolist()
                    entry['sphere_axes']=(float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())*relative[:3,:3]).tolist()
                self.entries.append(entry)
        assert self.entries
        self.ids=[x['body_id'] for x in self.entries]
        self.corners=torch.tensor(np.array([x['corners'] for x in self.entries]),device=env.device)
        manifest=json.loads((Path(__file__).resolve().parents[1]/'reference/native_kinematics.json').read_text())
        repo=usd.parents[5]
        # Resolve by exact known suffix instead of assuming repository depth.
        repo=Path(str(usd).split('/source/LeggedManip_Lab/')[0])
        self.hashes={k:dict(expected=v,actual=hashlib.sha256((repo/k).read_bytes()).hexdigest()) for k,v in manifest['layers'].items()}
        assert all(x['expected']==x['actual'] for x in self.hashes.values()),self.hashes

    def clearance(self,env):
        data=env.scene['robot'].data
        n=env.num_envs;m=len(self.ids)
        q=data.body_quat_w[:,self.ids,None,:].expand(n,m,8,4)
        p=data.body_pos_w[:,self.ids,None,:]
        xyz=self.quat_apply(q.reshape(-1,4),self.corners[None].expand(n,m,8,3).reshape(-1,3)).reshape(n,m,8,3)+p
        assert torch.isfinite(xyz).all(),'Non-finite world collision corners'
        lower=xyz[:,:,:,2].amin(2)
        for j,e in enumerate(self.entries):
            if 'sphere_center' in e:
                center=torch.tensor(e['sphere_center'],device=env.device,dtype=torch.float32).expand(n,3)
                axes=torch.tensor(e['sphere_axes'],device=env.device,dtype=torch.float32)[None].expand(n,3,3)
                rotation=data.body_quat_w[:,e['body_id']]
                c=self.quat_apply(rotation,center)[:,2]+data.body_pos_w[:,e['body_id'],2]
                az=self.quat_apply(rotation[:,None].expand(n,3,4).reshape(-1,4),axes.reshape(-1,3)).reshape(n,3,3)[:,:,2]
                lower[:,j]=c-az.norm(dim=-1)
        return lower.amin(1)-env.scene.env_origins[:,2]

