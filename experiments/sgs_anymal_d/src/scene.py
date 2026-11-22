"""Isaac Lab ANYmal-D goal navigation and non-training validation entry."""

from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import random
import time
import traceback

import navigation as nav
from pool import EVAL_STREAM, TRAIN_STREAM, task
from terrain import (GEOMETRY_VERSION, LEVELS, TERRAIN_NAMES, TERRAIN_SEEDS, TILE_COUNT, birth_lift,
                     decode, encode, mesh, tile)


TERRAIN_PRIM = "/World/SGSTerrain"
PHYSICS_DT = .005
CONTROL_SUBSTEPS = 4
LABELS = ("running", "success", "drop", "base_contact", "timeout")
BOX_BIRTH_IDS = (32729, 33929, 38046, 38446, 35529, 35929, 39646)


def stage_terrain(stage, combinations, origins):
    import numpy as np
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, Vt

    vertices, indices = [], []
    offset = 0
    for (kind, seed, level), (x, y, z) in zip(combinations, origins, strict=True):
        points, faces = mesh(tile(kind, seed, level).grid)
        world_points = np.asarray(points, dtype=np.float32)
        world_points += np.asarray((x,y,z), dtype=np.float32)
        vertices.append(world_points)
        indices.append(np.asarray(faces, dtype=np.int32).reshape(-1) + offset)
        offset += len(points)
    vertices = np.concatenate(vertices, axis=0)
    indices = np.concatenate(indices, axis=0)
    terrain = UsdGeom.Mesh.Define(stage, Sdf.Path(TERRAIN_PRIM))
    terrain.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    terrain.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(indices))
    terrain.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices)//3, 3, dtype=np.int32)))
    terrain.CreateSubdivisionSchemeAttr("none")
    terrain.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(.39, .52, .64)]))
    terrain.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(terrain.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(terrain.GetPrim()).CreateApproximationAttr("none")
    return dict(points=len(vertices), triangles=len(indices)//3, prim=TERRAIN_PRIM)


class GoalScene:
    def __init__(self, combinations, render, *, tile_bank=()):
        import omni.usd
        import torch
        import isaaclab.sim as sim_utils
        from isaaclab.assets import AssetBaseCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
        from isaaclab.utils import configclass
        from isaaclab_assets.robots.anymal import ANYMAL_D_CFG

        @configclass
        class SceneCfg(InteractiveSceneCfg):
            robot = ANYMAL_D_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
            contacts = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", update_period=0., history_length=4)
            scan = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/base",
                offset=RayCasterCfg.OffsetCfg(pos=(0.,0.,3.)),
                ray_alignment="yaw", update_period=nav.CONTROL_DT,
                pattern_cfg=patterns.GridPatternCfg(resolution=.10, size=(1.6,1.)),
                mesh_prim_paths=[TERRAIN_PRIM],
            )
            light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2300.))

        self.torch = torch
        self.device = "cuda:0"
        self.combinations = list(combinations)
        self.n = len(combinations)
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=PHYSICS_DT, render_interval=CONTROL_SUBSTEPS, device=self.device, use_fabric=True,
            physx=sim_utils.PhysxCfg(gpu_collision_stack_size=(256*2**20 if self.n==32768 else 2**26),
                                     gpu_max_rigid_patch_count=max(5*2**15, 512*self.n),
                                     gpu_max_rigid_contact_count=max(2**23, 1024*self.n))))
        self.scene = InteractiveScene(SceneCfg(num_envs=self.n, env_spacing=15., filter_collisions=bool(tile_bank),
                                               lazy_sensor_update=False))
        stage = omni.usd.get_context().get_stage()
        tile_bank = tuple(dict.fromkeys(tile_bank))
        bank_origins = [((i%17)*15.-120., (i//17)*15.-120., 0.) for i in range(len(tile_bank))]
        self.bank = dict(zip(tile_bank, bank_origins, strict=True))
        if self.bank:
            self.geometry = stage_terrain(stage, tile_bank, bank_origins)
        else:
            self.geometry = stage_terrain(stage, combinations, self.scene.env_origins.cpu().tolist())
        self.stage = stage
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.contact = self.scene["contacts"]
        self.scan = self.scene["scan"]
        self.origins = self.scene.env_origins.clone()
        self.env_ids = torch.arange(self.n, device=self.device)
        self.ids = [-1] * self.n
        self.steps = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.hold = self.steps.clone()
        self.targets = torch.zeros((self.n, 3), device=self.device)
        self.start_locations = self.targets.clone()
        self.last_action = torch.zeros((self.n, 12), device=self.device)
        self.previous_contact = torch.zeros((self.n, 4), dtype=torch.bool, device=self.device)
        self.last_reward = torch.zeros((self.n, 7), device=self.device)
        self.foot_bodies, self.foot_names = self.robot.find_bodies(".*_FOOT")
        self.foot_sensors, sensor_names = self.contact.find_bodies(self.foot_names, preserve_order=True)
        if list(sensor_names) != list(self.foot_names) or len(self.foot_bodies) != 4:
            raise RuntimeError((self.foot_names, sensor_names))
        self.base_sensor = self.contact.find_bodies("base")[0][0]
        self.non_feet = [i for i in range(len(self.contact.body_names)) if i not in self.foot_sensors]
        if self.robot.num_joints != 12 or self.robot.root_physx_view.get_root_transforms().device.type != "cuda":
            raise RuntimeError("Expected the 12-joint CUDA ANYmal-D articulation")
        self.rngs = {key: random.Random(value) for key, value in (("training",TRAIN_STREAM), ("evaluation",EVAL_STREAM))}
        self.render = render

    def reset_tasks(self, requests):
        """Reset selected slots from global SGS task IDs against the loaded terrain bank."""
        slots, ids = tuple(requests), tuple(requests.values())
        return self.assign(slots, ids)

    def assign(self, indices, task_ids, override=None):
        torch = self.torch
        if len(indices) != len(task_ids):
            raise ValueError("Need one task per robot")
        if not indices:
            return self.start_locations[:0].clone()
        ids = torch.tensor(indices, device=self.device, dtype=torch.long)
        if any(not 0 <= slot < self.n for slot in indices) or len(set(indices)) != len(indices):
            raise ValueError(f"Invalid reset slots {indices}")
        requested = [decode(task_id)[:3] for task_id in task_ids]
        if self.bank:
            if any(key not in self.bank for key in requested):
                raise ValueError(f"Task terrain absent from preloaded bank: {requested}")
        elif any(key != self.combinations[slot] for slot, key in zip(indices, requested, strict=True)):
            raise ValueError(f"Task terrain conflicts with an assigned fixed slot: {requested}")
        pairs = []
        for i, task_id, key in zip(indices, task_ids, requested, strict=True):
            self.ids[i] = task_id
            self.combinations[i] = key
            pairs.append(task(task_id))
        if self.bank:
            self.origins[ids] = torch.as_tensor([self.bank[key] for key in requested], device=self.device)
        starts = torch.as_tensor([a for a, _ in pairs], device=self.device, dtype=torch.float32)
        goals = torch.as_tensor([b for _, b in pairs], device=self.device, dtype=torch.float32)
        if override:
            for i, location in override.items():
                goals[i] = torch.as_tensor(location, device=self.device)
        self.scene.reset(ids)
        self.start_locations[ids] = starts + self.origins[ids]
        self.targets[ids] = goals + self.origins[ids]
        roots = self.robot.data.default_root_state[ids].clone()
        roots[:, :3] = self.start_locations[ids]
        roots[:, 2] += nav.BASE_HEIGHT + roots.new_tensor([birth_lift(task_id) for task_id in task_ids])
        self.robot.write_root_pose_to_sim(roots[:, :7], env_ids=ids)
        self.robot.write_root_velocity_to_sim(roots[:, 7:], env_ids=ids)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos[ids],
                                            self.robot.data.default_joint_vel[ids], env_ids=ids)
        self.last_action[ids] = 0.
        self.previous_contact[ids] = False
        self.steps[ids] = 0
        self.hold[ids] = 0
        actual = self.robot.root_physx_view.get_root_transforms()[ids, :3].clone()
        if not torch.allclose(actual, roots[:, :3], atol=1e-5, rtol=0.):
            raise RuntimeError("Native reset positions differ from the task library")
        return actual

    def draw(self, slot, split):
        rng = self.rngs[split]
        return encode(*self.combinations[slot], rng.randrange(20), rng.randrange(20))

    def ray_hits(self, positions, offset=2.):
        from isaaclab.utils.warp import raycast_mesh

        sources = positions.clone()
        sources[..., 2] += offset
        directions = self.torch.zeros_like(sources)
        directions[..., 2] = -1.
        hits, _, _, _ = raycast_mesh(sources, directions, self.scan.meshes[TERRAIN_PRIM])
        return hits

    def observations(self):
        from isaaclab.utils.math import quat_apply_inverse

        torch = self.torch
        bot = self.robot.data
        target = self.targets - bot.root_pos_w
        target[:, 2] += nav.BASE_HEIGHT
        goal = quat_apply_inverse(bot.root_quat_w, target)
        scan = self.scan.data.ray_hits_w[:, :, 2]
        height = torch.nan_to_num(bot.root_pos_w[:, 2:3] - scan - nav.BASE_HEIGHT, nan=0., posinf=2., neginf=-2.)
        return torch.cat((bot.root_lin_vel_b, bot.root_ang_vel_b, bot.projected_gravity_b,
                          bot.joint_pos - bot.default_joint_pos, bot.joint_vel, self.last_action, goal, height), dim=1)

    def step(self, action, *, training_iteration=0, after_physics=None):
        torch = self.torch
        if tuple(action.shape) != (self.n, 12):
            raise ValueError(action.shape)
        self.last_action.copy_(action)
        self.robot.set_joint_position_target(self.robot.data.default_joint_pos + nav.ACTION_SCALE * action)
        for substep in range(CONTROL_SUBSTEPS):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(PHYSICS_DT)
            if after_physics and substep==0:
                after_physics()
        state = self.robot.data
        root = state.root_pos_w
        xy = self.targets[:, :2] - root[:, :2]
        dist = xy.norm(dim=-1)
        z_error = (self.targets[:, 2] + nav.BASE_HEIGHT - root[:, 2]).abs()
        near = (dist <= nav.GOAL_RADIUS) & (z_error <= nav.GOAL_HEIGHT_TOLERANCE)
        self.steps += 1
        force = self.contact.data.net_forces_w.norm(dim=-1)
        max_force = self.contact.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
        foot = force[:, self.foot_sensors]
        foot_contact = foot > nav.MIN_CONTACT_FORCE
        from diagnostics import stable_arrival
        stable = stable_arrival(near, state.projected_gravity_b[:,2],
                                state.root_lin_vel_w[:,:2].norm(dim=-1),
                                state.root_ang_vel_w.norm(dim=-1), foot_contact.sum(dim=1))
        self.hold = torch.where(stable,self.hold+1,0)
        touchdown = foot_contact & ~self.previous_contact
        self.previous_contact.copy_(foot_contact)
        result = torch.zeros(self.n, dtype=torch.int8, device=self.device)
        result[self.steps >= nav.MAX_STEPS] = 4
        result[self.hold >= nav.SUCCESS_HOLD] = 1
        result[max_force[:, self.base_sensor] > nav.BASE_CONTACT_FORCE] = 3
        result[root[:, 2] - self.origins[:, 2] < nav.DROP_HEIGHT] = 2
        forward_xy = state.root_lin_vel_b[:, :2]
        from isaaclab.utils.math import quat_apply_inverse
        goal_vector = torch.column_stack((xy, torch.zeros_like(dist)))
        goal_body = quat_apply_inverse(state.root_quat_w, goal_vector)[:, :2]
        align = ((forward_xy * goal_body).sum(dim=1) /
                 (forward_xy.norm(dim=1).clamp_min(1e-6) * goal_body.norm(dim=1).clamp_min(1e-6))).clamp_min(0.)
        vel_feet = state.body_lin_vel_w[:, self.foot_bodies, 2]
        terms = torch.stack((result == 1, (result == 2) | (result == 3),
                             (state.applied_torque * state.joint_vel).abs().sum(dim=1),
                             (state.joint_pos - state.default_joint_pos).abs().sum(dim=1),
                             (touchdown * vel_feet.square()).sum(dim=1),
                             (max_force[:, self.non_feet] > nav.MIN_CONTACT_FORCE).sum(dim=1), align), dim=1)
        self.last_reward.copy_(terms)
        if not torch.isfinite(root).all() or not torch.isfinite(terms).all():
            raise RuntimeError("Invalid navigation tensors")
        return result, dict(distance=dist, height_error=z_error, near=near, foot_force=foot,
                            base_force=max_force[:, self.base_sensor],
                            reward=terms @ terms.new_tensor(nav.reward_weights(training_iteration)))

    def row(self, i, result, state):
        return dict(slot=i, task_id=self.ids[i], terrain=TERRAIN_NAMES[self.combinations[i][0]],
                    seed=self.combinations[i][1], level=self.combinations[i][2], age=int(self.steps[i]),
                    label=LABELS[int(result[i])], hold=int(self.hold[i]),
                    start_world=self.start_locations[i].tolist(), target_world=self.targets[i].tolist(),
                    root_world=self.robot.data.root_pos_w[i].tolist(), foot_force_n=state["foot_force"][i].tolist(),
                    base_force_n=float(state["base_force"][i]), distance_m=float(state["distance"][i]),
                    height_error_m=float(state["height_error"][i]), reward=float(state["reward"][i]),
                    reward_terms=self.last_reward[i].tolist(), action=self.last_action[i].tolist())


def engineering_action(world, tick):
    torch = world.torch
    phase = torch.arange(12, device=world.device).view(1, -1) * .91
    terrain_offset = torch.arange(world.n, device=world.device).view(-1,1) * .15
    return .14 * torch.sin(tick * .18 + phase + terrain_offset)


def sample(world, i, result, state):
    row = world.row(i, result, state)
    obs = world.observations()
    rays = world.scan.data.ray_hits_w[i, :,2]
    finite = rays[world.torch.isfinite(rays)]
    assert finite.numel() and world.torch.isfinite(obs[i]).all()
    row.update(scan_rays=rays.numel(), scan_hits=finite.numel(), scan_height_min=float(finite.min()),
               scan_height_max=float(finite.max()),
               observation_width=obs.shape[1])
    return row


def representative(world):
    torch = world.torch
    slots = list(range(world.n))
    ids = [world.draw(i, "training") for i in slots]
    train_state = world.rngs["training"].getstate()
    eval_ids = [world.draw(i, "evaluation") for i in slots]
    assert train_state == world.rngs["training"].getstate() and ids != eval_ids
    initial = world.assign(slots, ids)
    starts, goals = world.start_locations.clone(), world.targets.clone()
    t0 = time.perf_counter()
    logs, terminals = [], []
    labels = Counter()
    asynchronous = None
    force_max = 0.
    executed = 525
    for tick in range(executed):
        result, signals = world.step(engineering_action(world, tick))
        force_max = max(force_max, float(signals["foot_force"].max()))
        if tick in {0, 10, 25, 100, 300, 500}:
            logs.extend(sample(world, i, result, signals) for i in slots)
        ended = result.nonzero().flatten().tolist()
        for i in ended:
            record = world.row(i, result, signals)
            assert nav.terminal(record["root_world"][2]-float(world.origins[i,2]), record["base_force_n"],
                                record["hold"], record["age"]) == record["label"]
            labels[record["label"]] += 1
            terminals.append(record)
        if ended:
            for i in ended:
                before = world.robot.root_physx_view.get_root_transforms().clone()
                old_ids = world.ids.copy()
                world.assign([i], [world.draw(i, "training")])
                others = [j for j in slots if j != i]
                neighbours_held = bool(torch.equal(before[others], world.robot.root_physx_view.get_root_transforms()[others]))
                assert neighbours_held and [old_ids[j] for j in others] == [world.ids[j] for j in others]
                if asynchronous is None:
                    asynchronous = dict(tick=tick+1, reset_slot=i, untouched_slots=others, native_exact=True,
                                        previous_id=old_ids[i], new_id=world.ids[i])
    elapsed = time.perf_counter() - t0
    assert asynchronous and force_max > nav.MIN_CONTACT_FORCE

    # Use the same task IDs and held actions across two real simulator replays.
    trajectories = []
    for _ in range(2):
        reset = world.assign(slots, ids)
        assert torch.equal(initial, reset)
        assert torch.equal(starts,world.start_locations) and torch.equal(goals,world.targets)
        trace = []
        for tick in range(12):
            result, _ = world.step(engineering_action(world,tick))
            trace.append(world.robot.data.root_pos_w.clone())
        trajectories.append(torch.stack(trace))
    replay_error = (trajectories[0]-trajectories[1]).abs().max().item()
    if replay_error > .02:
        raise RuntimeError(f"Replay drift {replay_error} m exceeds 2 cm")

    # Place one controlled goal on its saved, supported starting location.
    # Other slots retain their real library goals and provide negative controls.
    positive = 0
    start = task(ids[positive])[0]
    world.assign(slots, ids, override={positive:start})
    controlled = []
    for tick in range(nav.SUCCESS_HOLD):
        result, signals = world.step(torch.zeros_like(world.last_action))
        controlled.append([world.row(i,result,signals) for i in slots])
    assert controlled[-1][positive]["label"] == "success", controlled[-1][positive]
    assert all(row["label"] != "success" for row in controlled[-1][1:])
    assert all(step[positive]["label"] != "success" for step in controlled[:-1])
    before = world.robot.root_physx_view.get_root_transforms().clone()
    continuing = [i for i in slots if i != positive]
    preserved_goals = world.targets[continuing].clone()
    preserved_ids = [world.ids[i] for i in continuing]
    world.assign([positive], [world.draw(positive, "training")])
    assert torch.equal(before[continuing], world.robot.root_physx_view.get_root_transforms()[continuing])
    assert torch.equal(preserved_goals, world.targets[continuing])
    assert preserved_ids == [world.ids[i] for i in continuing]
    assert world.steps.tolist() == [0] + [nav.SUCCESS_HOLD] * len(continuing)
    next_result, next_signals = world.step(torch.zeros_like(world.last_action))
    after_partial = [world.row(i, next_result, next_signals) for i in slots]
    assert [row["age"] for row in after_partial] == [1] + [nav.SUCCESS_HOLD + 1] * len(continuing)
    return dict(control_steps=executed, environment_transitions=executed*world.n, seconds=elapsed,
                effective_transitions_per_s=executed*world.n/elapsed,
                seed_replay_max_xyz_error_m=replay_error, eval_rng_isolated=True, evaluated_task_ids=eval_ids,
                natural_labels=dict(labels), natural_terminals=terminals[:40], sampled=logs,
                maximum_foot_force_n=force_max, asynchronous_reset=asynchronous,
                controlled_final=controlled[-1], controlled_first_step=controlled[0],
                controlled_post_asynchronous_reset=after_partial)


def coverage(world):
    ids = [world.draw(i, "evaluation") for i in range(world.n)]
    world.assign(list(range(world.n)), ids)
    anchors = world.torch.cat((world.start_locations, world.targets), dim=0)
    verified = world.ray_hits(anchors)
    errors = (verified[:, 2] - anchors[:, 2]).abs()
    assert world.torch.isfinite(errors).all() and errors.max() < .0001
    maximum = world.torch.zeros(world.n, device=world.device)
    t0 = time.perf_counter()
    for tick in range(28):
        result, state = world.step(engineering_action(world,tick))
        maximum = world.torch.maximum(maximum, state["foot_force"].amax(dim=1))
    elapsed = time.perf_counter() - t0
    rows = [sample(world,i,result,state) for i in range(world.n)]
    for i, (row, largest) in enumerate(zip(rows,maximum.tolist())):
        row["max_foot_force_n"] = largest
        row["start_support_height_error_m"] = float(errors[i])
        row["goal_support_height_error_m"] = float(errors[i+world.n])
    return dict(coverage=rows, control_steps=28, seconds=elapsed,
                effective_transitions_per_s=28*world.n/elapsed)


def handoff(world):
    torch = world.torch
    route = [
        encode(0,0,1,7,2), encode(10,1,2,3,1),
        encode(3,1,8,1,6), encode(4,0,4,7,18),
        encode(10,0,9,5,3),
    ]
    held = [encode(9,0,4,4,6), encode(7,1,4,2,17)]
    world.reset_tasks({0:route[0], 1:held[0], 2:held[1]})
    bank_support = []
    neighbours = world.robot.root_physx_view.get_root_transforms()[1:].clone()
    time_step = world.sim.current_time_step_index
    bank_t0 = time.perf_counter()
    for layout, origin in world.bank.items():
        chosen = encode(*layout, 0, 0)
        pose = world.reset_tasks({0:chosen})
        sheet = tile(*layout)
        bank_support.extend([tuple(p+d for p,d in zip(point,origin,strict=True)) for point in sheet.starts+sheet.goals])
        assert torch.equal(pose[0,:2], world.start_locations[0,:2])
    assert torch.equal(neighbours, world.robot.root_physx_view.get_root_transforms()[1:])
    assert time_step == world.sim.current_time_step_index and world.ids[1:] == held
    all_anchors = torch.as_tensor(bank_support, dtype=torch.float32, device=world.device)
    bank_ray_error = (world.ray_hits(all_anchors)-all_anchors).abs().max().item()
    assert bank_ray_error < .0001
    bank_elapsed = time.perf_counter()-bank_t0
    track = []
    for selected in route:
        before = world.robot.root_physx_view.get_root_transforms()[1:].clone()
        before_goals = world.targets[1:].clone()
        before_age = world.steps[1:].clone()
        actual = world.reset_tasks({0:selected})
        assert torch.equal(before, world.robot.root_physx_view.get_root_transforms()[1:])
        assert torch.equal(before_goals, world.targets[1:]) and world.ids[1:] == held
        assert torch.equal(before_age, world.steps[1:])
        anchors = torch.stack((world.start_locations[0],world.targets[0]))
        error = (world.ray_hits(anchors) - anchors).abs().max().item()
        assert error < .0001
        max_contact = 0.
        for tick in range(26):
            outcomes, measurements = world.step(engineering_action(world,tick))
            max_contact = max(max_contact,float(measurements["foot_force"][0].max()))
        row = sample(world,0,outcomes,measurements)
        row.update(placement_xyz=actual[0].tolist(), anchor_height_error_m=error,
                   unaffected_slots=[1,2], continuing_ages=world.steps[1:].tolist(),
                   continued_for=int((world.steps[1:]-before_age).min()), maximum_contact_n=max_contact)
        assert row["continued_for"] == 26 and row["maximum_contact_n"] > nav.MIN_CONTACT_FORCE, row
        track.append(row)

    # Two robots deliberately get identical terrain, task ID and birth pose.
    # Slot 1 receives vigorous action while the fixed slot 0 matches its isolated trace.
    repeated = route[0]
    trajectories = []
    for colocated in (False, True):
        second = repeated if colocated else held[0]
        born = world.reset_tasks({0:repeated, 1:second, 2:held[1]})
        assert bool(torch.equal(born[0], born[1])) == colocated
        trace = []
        for tick in range(24):
            actions = torch.zeros_like(world.last_action)
            actions[1] = engineering_action(world,tick)[1] * 4.
            _, signals = world.step(actions)
            trace.append(torch.cat((world.robot.data.root_pos_w[0], signals["foot_force"][0])).clone())
        trajectories.append(torch.stack(trace))
    difference = (trajectories[0]-trajectories[1]).abs().amax(dim=0)
    assert float(difference[:3].max()) < .003 and float(difference[3:].max()) < 5., difference
    return dict(route=track, selectable_tiles=len(world.bank), selectable_task_ids=400*len(world.bank),
                native_slot_resets_across_bank=len(world.bank), verified_bank_patch_count=len(bank_support),
                bank_max_height_error_m=bank_ray_error, bank_resets_and_rays_wall_s=bank_elapsed,
                full_catalogue_available=len(world.bank)==TILE_COUNT,
                smaller_scene_held_slots=held, same_task_same_birth=True,
                isolation_control_steps=24, isolation_position_error_m=float(difference[:3].max()),
                isolation_foot_force_error_n=float(difference[3:].max()))


def birth_diagnostics(world):
    torch = world.torch
    tasks = dict(enumerate(BOX_BIRTH_IDS))
    observations = {}
    for strategy in ("original", "zero_action"):
        world.reset_tasks(tasks)
        world.sim.forward()
        feet = world.robot.root_physx_view.get_link_transforms()[:, world.foot_bodies, :3].clone()
        gap = feet[:, :, 2]-world.ray_hits(feet)[:, :, 2]
        assert torch.isfinite(gap).all()
        heights = world.ray_hits(feet)[:, :, 2].tolist()
        traces = [[] for _ in tasks]
        first_foot, first_other, first_terminal = {}, {}, {}
        max_foot = torch.zeros((world.n,4),device=world.device)
        for tick in range(120):
            actions = torch.zeros_like(world.last_action)
            if strategy == "original":
                phase = torch.arange(12,device=world.device)[None,:] * .91
                old_slot = torch.as_tensor([8+decode(task_id)[1] for task_id in BOX_BIRTH_IDS],
                                           device=world.device)[:,None]
                actions = .14*torch.sin(tick*.18+phase+old_slot*.15)
            results, measurements = world.step(actions)
            all_forces = world.contact.data.net_forces_w_history.norm(dim=-1).amax(dim=1)
            max_foot = torch.maximum(max_foot, all_forces[:,world.foot_sensors])
            for slot, task_id in tasks.items():
                foot = [name for name, force in zip(world.foot_names, all_forces[slot,world.foot_sensors].tolist())
                        if force > nav.MIN_CONTACT_FORCE]
                other = [world.contact.body_names[i] for i in world.non_feet
                         if float(all_forces[slot,i]) > nav.MIN_CONTACT_FORCE]
                if foot and task_id not in first_foot:
                    first_foot[task_id] = dict(step=tick+1, feet=foot)
                if other and task_id not in first_other:
                    first_other[task_id] = dict(step=tick+1, bodies=other)
                if int(results[slot]) and task_id not in first_terminal:
                    first_terminal[task_id] = dict(step=tick+1,label=LABELS[int(results[slot])])
                if tick+1 in (1,4,8,14,28,42,60,120):
                    traces[slot].append(dict(world.row(slot,results,measurements),contacting_feet=foot,
                                             contacting_non_feet=other))
        observations[strategy] = [
            dict(task_id=task_id, original_geometry=world.combinations[slot], initial_feet_world=feet[slot].tolist(),
                 floor_under_each_foot_m=heights[slot], initial_foot_center_clearance_m=gap[slot].tolist(),
                 contacts_through_120_max_n=max_foot[slot].tolist(), first_foot=first_foot.get(task_id),
                 first_non_foot=first_other.get(task_id), first_terminal=first_terminal.get(task_id), trace=traces[slot])
            for slot,task_id in tasks.items()]
    return dict(configurations=list(BOX_BIRTH_IDS), control_steps_per_strategy=120, strategies=observations)


def marker(world, xyz):
    from pxr import Gf, UsdGeom, Vt
    sphere = UsdGeom.Sphere.Define(world.stage, "/World/GoalMarker")
    sphere.CreateRadiusAttr(.12)
    sphere.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(.88,.13,.06)]))
    UsdGeom.Xformable(sphere).AddTranslateOp().Set(Gf.Vec3d(*xyz))


def render_frames(world, folder):
    import omni.replicator.core as rep
    from PIL import Image

    ids = [world.draw(i, "evaluation") for i in range(world.n)]
    world.assign(list(range(world.n)), ids)
    foot = world.start_locations[0].tolist()
    goal = world.targets[0].tolist()
    marker(world, (goal[0], goal[1], goal[2]+.12))
    focus = ((foot[0]+goal[0])/2, (foot[1]+goal[1])/2, max(foot[2],goal[2])+.3)
    camera = rep.create.camera(position=(focus[0]+6., focus[1]-7., focus[2]+6.), look_at=focus)
    product = rep.create.render_product(camera, (960, 600))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product])
    outputs = []
    for k in range(3):
        for step in range(8):
            world.step(engineering_action(world,k*8+step))
        rep.orchestrator.step(rt_subframes=5, delta_time=0.)
        data = rgb.get_data()
        path = folder / f"frame_{k:02}.png"
        Image.fromarray(data[:, :, :3]).save(path)
        outputs.append(dict(path=str(path), width=data.shape[1], height=data.shape[0]))
    rgb.detach()
    return dict(frames=outputs, goal=goal, initial_base=foot)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=("representative","coverage","render","frames","handoff","handoff_full","birth"),
                        required=True)
    parser.add_argument("--level", type=int, choices=range(LEVELS), default=0)
    parser.add_argument("--experience", required=True)
    parser.add_argument("--kit-args", required=True)
    args = parser.parse_args()
    from isaaclab.app import AppLauncher
    from tools.p3s2_frozen_protocol import launcher_class

    render = args.mode == "render"
    app = launcher_class(AppLauncher,1)(headless=True, device="cuda:0", enable_cameras=render,livestream=0,
                                        experience=args.experience, kit_args=args.kit_args).app
    try:
        bank = ()
        if args.mode == "coverage":
            combinations = [(kind,seed,args.level) for kind in range(len(TERRAIN_NAMES)) for seed in range(2)]
        elif render or args.mode == "frames":
            combinations = [(10,1,2)]
        elif args.mode in ("handoff","handoff_full"):
            combinations = [(0,0,1),(9,0,4),(7,1,4)]
            bank = [*combinations, (10,1,2),(3,1,8),(4,0,4),(10,0,9)]
            if args.mode == "handoff_full":
                bank = [(kind,seed,level) for kind in range(len(TERRAIN_NAMES)) for seed in range(TERRAIN_SEEDS)
                        for level in range(LEVELS)]
        elif args.mode == "birth":
            combinations = [decode(task_id)[:3] for task_id in BOX_BIRTH_IDS]
            bank = combinations
        else:
            combinations = [(0,0,1),(3,1,6),(9,0,4),(10,1,2)]
        world = GoalScene(combinations, render, tile_bank=bank)
        if args.mode == "representative":
            result = representative(world)
        elif args.mode == "coverage":
            result = coverage(world)
        elif args.mode in ("handoff","handoff_full"):
            result = handoff(world)
        elif args.mode == "birth":
            result = birth_diagnostics(world)
        elif args.mode == "frames":
            from render_source import measured_frames

            result = measured_frames(world,args.output,world.draw(0,"evaluation"),engineering_action)
        else:
            result = render_frames(world,args.output)
        receipt = dict(mode=args.mode, geometry_version=GEOMETRY_VERSION, parallel=len(combinations), combinations=combinations,
                       scene_geometry=world.geometry, torch_device=world.device, action_width=world.robot.num_joints,
                       ray_size=list(world.scan.data.ray_hits_w.shape), feet=world.foot_names,
                       physics_dt=PHYSICS_DT, control_dt=nav.CONTROL_DT, ppo_updates=0, **result)
        (args.output / "scene_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    except BaseException:
        (args.output / "scene_error.txt").write_text(traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__ == "__main__":
    main()
