from functools import partial
import numpy as np
import torch
import navigation as nav
from catalog import ROOT
from scene import GoalScene

class BatchedScene(GoalScene):
    def __init__(self,count):
        layouts=[(a,b,c) for a in range(13) for b in range(2) for c in range(10)]
        super().__init__([layouts[i%260] for i in range(count)],False,tile_bank=layouts)
        self.lookup=torch.as_tensor(np.load(ROOT/'geometry.npy'),device=self.device)
        self.atlas_offsets=torch.tensor([self.bank[key] for key in layouts],device=self.device)
        self.task_ids=torch.full((count,),-1,device=self.device,dtype=torch.long)
    def place(self,slots,ids,verify=False):
        self.scene.reset(slots)
        g=self.lookup[ids]; offsets=self.atlas_offsets[ids//400]
        self.task_ids[slots]=ids; self.origins[slots]=offsets
        self.start_locations[slots]=g[:,:3]+offsets; self.targets[slots]=g[:,3:6]+offsets
        roots=self.robot.data.default_root_state[slots].clone()
        roots[:,:3]=self.start_locations[slots];roots[:,2]+=nav.BASE_HEIGHT+g[:,6]
        # A start pose is independent of the goal. Face the obstacle from either apron.
        yaw = torch.where(g[:,0]<0,0.,torch.pi) + .20*torch.sin(13*g[:,0]+7*g[:,1])
        roots[:,3:7]=torch.stack((torch.cos(yaw/2),torch.zeros_like(yaw),torch.zeros_like(yaw),torch.sin(yaw/2)),dim=1)
        self.robot.write_root_pose_to_sim(roots[:,:7],env_ids=slots)
        self.robot.write_root_velocity_to_sim(roots[:,7:],env_ids=slots)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[slots],self.robot.data.default_joint_vel[slots],env_ids=slots)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos[slots],env_ids=slots)
        self.last_action[slots]=0.;self.previous_contact[slots]=False;self.steps[slots]=0;self.hold[slots]=0
        if verify:
            actual=self.robot.root_physx_view.get_root_transforms()[slots,:3]
            torch.testing.assert_close(actual,roots[:,:3],atol=1e-5,rtol=0.)
        self.sim.forward()
    def probe_geometry(self):
        maxerr=0.
        for begin in range(0,260,self.n):
            ids=torch.arange(begin,min(260,begin+self.n),device=self.device)*400+27
            slots=torch.arange(len(ids),device=self.device)
            self.place(slots,ids,verify=True)
            endpoints=torch.cat((self.start_locations[slots],self.targets[slots]))
            maxerr=max(maxerr,(self.ray_hits(endpoints)-endpoints).abs().max().item())
        if not maxerr<.00015: raise ValueError(('terrain-ray mismatch',maxerr))
        from pxr import PhysxSchema
        px=PhysxSchema.PhysxSceneAPI(self.stage.GetPrimAtPath(self.sim.cfg.physics_prim_path)); cfg=self.sim.cfg.physx
        active=dict(gpu_collision_stack_size=int(px.GetGpuCollisionStackSizeAttr().Get()),
                    gpu_max_rigid_patch_count=int(px.GetGpuMaxRigidPatchCountAttr().Get()),
                    gpu_max_rigid_contact_count=int(px.GetGpuMaxRigidContactCountAttr().Get()))
        for key,actual in active.items():assert getattr(cfg,key)==actual,(key,actual,getattr(cfg,key))
        if self.n==32768:assert active['gpu_collision_stack_size']==268435456
        return dict(physics_scene_prim=self.sim.cfg.physics_prim_path,solver_buffers=active,native_slots=self.robot.root_physx_view.get_root_transforms().shape[0],layouts=260,ray_xyz_max_error_m=maxerr,cuda_tensor=self.robot.root_physx_view.get_root_transforms().device.type,scan_rays=self.scan.data.ray_hits_w.shape[1])
