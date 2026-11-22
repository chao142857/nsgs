"""Read GPU contact points filtered to the robot's own collision plane."""


class NativeContacts:
    def __init__(self, rt, env):
        import torch
        from isaacsim.core.simulation_manager import SimulationManager

        self.rt=rt
        self.env=env
        self.torch=torch
        self.rows={}
        self.views={}
        view=SimulationManager.get_physics_sim_view()
        slope=rt.alpha[rt.state.task]
        mu=rt.mu[rt.state.task]
        roots=env.scene['robot'].root_physx_view.link_paths
        for name,mask in conditions(slope,mu):
            ids=torch.where(mask)[0][:6].tolist()
            assert ids,name
            for slot in ids:
                paths=[path for path in roots[slot] if path.endswith('_foot')]
                assert len(paths)==4,paths
                ground=rt.planes.view.prim_paths[slot]
                # Every foot reports against the collision plane of its own environment.
                pairs=[view.create_rigid_contact_view(path,[ground],max_contact_data_count=24) for path in paths]
                assert all(item.sensor_count==1 and item.filter_count==1 for item in pairs)
                self.views[slot]=list(zip(paths,pairs))

    def sample(self, control, previous_generation):
        from isaaclab.utils.math import quat_apply

        torch=self.torch
        rt=self.rt
        slope=rt.alpha[rt.state.task]
        mu=rt.mu[rt.state.task]
        stable=previous_generation==rt.state.generation
        for name,mask in conditions(slope,mu):
            if name in self.rows:
                continue
            for slot in (i for i in (mask&stable).nonzero().flatten().tolist() if i in self.views):
                origin=self.env.scene.env_origins[slot]
                normal=quat_apply(rt.plane_rotations[rt.state.task[slot]],origin.new_tensor((0.,0.,1.)))
                for path,pair in self.views[slot]:
                    forces,points,_,separations,counts,starts=pair.get_contact_data(self.env.physics_dt)
                    count=int(counts[0,0]); start=int(starts[0,0])
                    assert 0<=count<=24
                    if not count:
                        continue
                    assert 0<=start and start+count<=24
                    records=slice(start,start+count)
                    positions=points[records].to(origin.device)
                    residual=((positions-origin)@normal).abs()
                    forces=forces[records].reshape(-1).to(origin.device)
                    separation=separations[records].reshape(-1).to(origin.device)
                    height=(positions[:,2]-origin[2]).abs()
                    hits=torch.where((forces>1.) & (separation.abs()<.006) & (residual<.008) & (height>.018))[0]
                    if len(hits):
                        j=int(hits[0]); point=positions[j]
                        self.rows[name]=dict(control=control,slot=slot,alpha_deg=float(slope[slot]),mu=float(mu[slot]),
                            foot_prim=path,plane_prim=rt.planes.view.prim_paths[slot],
                            force_n=float(forces[j]),separation_m=float(separation[j]),point_world_m=point.tolist(),
                            slope_residual_m=float(residual[j]),flat_residual_m=float(height[j]))
                        break
                if name in self.rows:
                    break


def conditions(slope,mu):
    return (('negative_low_mu',(slope<-10)&(mu<.56)),
            ('negative_high_mu',(slope<-10)&(mu>.85)),
            ('positive_low_mu',(slope>10)&(mu<.56)),
            ('positive_high_mu',(slope>10)&(mu>.85)))


def gpu_plane_intersections(rt, env, actual, slots):
    import math
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from protocol import PLATFORM_METRES

    rows=[]
    for slot in sorted(slots):
        pose=actual[slot].to(env.device)
        center=pose[:3]
        rotation=pose[[6,3,4,5]]
        normal=quat_apply(rotation,pose.new_tensor((0.,0.,1.)))
        origin=env.scene.env_origins[slot]
        thickness=PLATFORM_METRES[2]
        top=center+normal*(.5*thickness)
        alpha=float(rt.alpha[rt.state.task[slot]])
        radians=math.radians(alpha)
        expected=(-math.sin(radians),0.,math.cos(radians))
        assert max(abs(float(a)-b) for a,b in zip(normal,expected))<.001
        for dx in (-1.6,1.6):
            ray_start=origin+pose.new_tensor((dx,.7,2.))
            distance=((ray_start-top)@normal)/normal[2]
            point=ray_start-pose.new_tensor((0.,0.,1.))*distance
            local=quat_apply_inverse(rotation,point-center)
            assert abs(float(local[0]))<=.5*PLATFORM_METRES[0] and abs(float(local[1]))<=.5*PLATFORM_METRES[1]
            assert abs(float(point[2]-origin[2])-dx*math.tan(radians))<.002
            rows.append(dict(slot=slot,alpha_deg=alpha,dx=dx,height=float(point[2]-origin[2]),normal=normal.tolist()))
    return rows
