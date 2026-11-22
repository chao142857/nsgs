"""Read-only CPU measurements; does not draw RNG or change simulator state."""
import torch


def conditions(env, ids):
    robot=env.scene['robot']; rt=env.goal_runtime
    index=ids.cpu()
    mass=robot.root_physx_view.get_masses()[index]
    materials=robot.root_physx_view.get_material_properties()[index]
    base_id=robot.find_bodies('base')[0][0]
    com=robot.data.body_com_pos_b[ids,base_id]
    rows=[]
    for k,i in enumerate(ids.cpu().tolist()):
        actuators={}
        for name,a in robot.actuators.items():
            def values(v):
                if not isinstance(v,torch.Tensor):return [float(v)]
                if v.ndim>1 and v.shape[0]==env.num_envs:v=v[i]
                return v.flatten().cpu().tolist()
            lag=a.positions_delay_buffer.time_lags
            actuators[name]=dict(stiffness=values(a.stiffness),damping=values(a.damping),
                delay_physics_steps=int(lag[i]))
        rows.append(dict(total_mass_kg=float(mass[k].sum()),base_mass_kg=float(mass[k,base_id]),
            base_com_m=com[k].cpu().tolist(),
            material_mean=materials[k].mean(0).cpu().tolist(),
            material_min=materials[k].amin(0).cpu().tolist(),
            material_max=materials[k].amax(0).cpu().tolist(),actuators=actuators,
            initial_joint_delta=rt.deltas[rt.state.task[i]].cpu().tolist()))
    return rows


def terminal_details(env, ids):
    rt=env.goal_runtime; r=env.scene['robot'].data
    sensor=env.scene.sensors['contact_forces']
    forces=sensor.data.net_forces_w_history.norm(dim=-1).amax(1)
    result=[]
    for i in ids.cpu().tolist():
        terms={name:float(v[i]) for name,v in env.reward_manager._episode_sums.items()}
        result.append(dict(episode_return=sum(terms.values()),reward_terms=terms,
            root_position_m=(r.root_pos_w[i]-env.scene.env_origins[i]).cpu().tolist(),
            root_quaternion_wxyz=r.root_quat_w[i].cpu().tolist(),
            end_effector_m=(r.body_pos_w[i,rt.ee]-env.scene.env_origins[i]).cpu().tolist(),
            nonfoot_contacts_n={sensor.body_names[j]:float(forces[i,j]) for j in rt.nonfoot if forces[i,j]>5},
            joint_position_rad=r.joint_pos[i,rt.action_ids].cpu().tolist()))
    return result
