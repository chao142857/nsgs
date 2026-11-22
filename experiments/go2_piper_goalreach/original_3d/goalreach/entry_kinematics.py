"""Full native joint tree used only for equilibrium reset geometry and audits."""
import json
from pathlib import Path
import numpy as np
from .kinematics import Arm


def rotation(q):
    w,x,y,z=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


class RobotTree:
    def __init__(self,path):
        self.joints=json.loads(Path(path).read_text())['joints']

    def poses(self,root_pose,q,names):
        q=np.atleast_2d(q);root_pose=np.broadcast_to(root_pose,(len(q),7))
        root=np.broadcast_to(np.eye(4),(len(q),4,4)).copy()
        root[:,:3,:3]=np.array([rotation(v) for v in root_pose[:,3:]])
        root[:,:3,3]=root_pose[:,:3]
        result={'/go2_piper/base':root};todo=list(self.joints)
        columns={name:i for i,name in enumerate(names)}
        while todo:
            ready=[j for j in todo if j['body0'] in result]
            if not ready:raise ValueError('Unresolved native joint tree')
            for j in ready:
                r=np.broadcast_to(np.eye(4),root.shape).copy()
                if j['kind']=='PhysicsRevoluteJoint':
                    angle=q[:,columns[j['name']]] if j['name'] in columns else np.zeros(len(q))
                    a='XYZ'.index(j['axis']);b,c=(a+1)%3,(a+2)%3
                    cs,sn=np.cos(angle),np.sin(angle)
                    r[:,b,b]=r[:,c,c]=cs;r[:,b,c]=-sn;r[:,c,b]=sn
                result[j['body1']]=result[j['body0']]@np.array(j['T0'])@r@np.linalg.inv(j['T1'])
                todo.remove(j)
        return result

    def reset_geometry(self,reference,delta,clearance=.001):
        names=reference['action_joint_names'];nominal=np.array(reference['joint_position'])
        q=nominal+delta;byname={j['name']:j for j in self.joints}
        for i,name in enumerate(names):
            j=byname[name];q[:,i]=np.clip(q[:,i],j['lower'],j['upper'])
        roots=np.broadcast_to(reference['root_pose'],(len(q),7)).copy()
        poses=self.poses(roots,q,names)
        # Native USD foot collision spheres, verified again by CollisionAudit.
        center=np.array([-.002,0,0]);radius=.022
        lower=np.stack([(poses['/go2_piper/'+leg+'_foot'][:,:3,:3]@center+
                         poses['/go2_piper/'+leg+'_foot'][:,:3,3])[:,2]-radius
                        for leg in ('FL','FR','RL','RR')],axis=1).min(1)
        lift=np.maximum(0.,clearance-lower);roots[:,2]+=lift
        ee=poses['/go2_piper/end_effector'][:,:3,3].copy();ee[:,2]+=lift
        return roots,q,ee,lift


class EntryArm(Arm):
    def __init__(self,path,reference):
        self.root_pose=None
        super().__init__(path)
        self.root_pose=np.array(reference['root_pose'])
        self.reference_q=np.array(reference['joint_position'][-6:])
        self.nominal=self.fk(self.reference_q)[0]

    def fk(self,q,return_links=False):
        points=super().fk(q,return_links)
        if self.root_pose is None:return points
        return (points-np.array([0,0,.35]))@rotation(self.root_pose[3:]).T+self.root_pose[:3]

    def admissible(self,q):
        links=self.fk(q,True);xyz=links[:,-1]
        samples=np.concatenate([links[:,2:-1]*(1-t)+links[:,3:]*t for t in np.linspace(0,1,9)],axis=1)
        body=(samples-self.root_pose[:3])@rotation(self.root_pose[3:])
        chassis=(abs(body[...,0])<.32)&(abs(body[...,1])<.20)&(body[...,2]<.13)
        return ((xyz[:,2]>=.10)&(xyz[:,2]<=.85)&(samples[...,2]>=.07).all(1)&~chassis.any(1))
