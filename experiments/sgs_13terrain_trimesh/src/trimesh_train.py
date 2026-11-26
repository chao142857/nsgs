"""Unified multi-baseline trainer on the 1:1 3D Trimesh 13-terrain benchmark.

Supports all 8 samplers from nsgs.baselines:
  uniform, tabular_sgs, grid_sgs, knn_sgs, plr, neural_plr, nsgs, nsgs_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from time import monotonic
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
REPO_SRC = EXP_ROOT.parent.parent / "src"
NSGS_SRC = EXP_ROOT / "nsgs_src"
sys.path.insert(0, str(HERE))
if REPO_SRC.is_dir():
    sys.path.insert(0, str(REPO_SRC))
if NSGS_SRC.is_dir():
    sys.path.insert(0, str(NSGS_SRC))

from trimesh_terrains import TERRAIN_NAMES

STEPS = 24  # PPO rollout horizon per update (matches SGS / v3)


def atomic_json(path: Path, data: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict):
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def load_task_pool(catalog_dir: Path):
    from nsgs.baselines.base import TaskPool
    meta = json.loads((catalog_dir / "catalog_meta.json").read_text())
    raw = np.load(catalog_dir / "raw_features.npy", allow_pickle=False).astype(np.float64)
    feat = np.load(catalog_dir / "model_features.npy", allow_pickle=False).astype(np.float32)
    bounds = tuple((float(lo), float(hi)) for lo, hi in meta["bounds"])
    periodic = tuple(int(d) for d in meta["periodic_dims"])
    pool = TaskPool(raw_features=raw, features=feat, bounds=bounds, periodic_dims=periodic, metadata=meta)
    return pool, meta


def make_sampler(method: str, pool, meta: dict, *, seed: int, device: str):
    from nsgs.baselines import SGSKernel, build_sampler
    if method == "grid_sgs":
        return build_sampler("grid_sgs", pool, seed=seed, bins=tuple(meta["grid_bins"]), kernel="sharp", history=100)
    if method in ("tabular_sgs", "tabular_sgs_sharp"):
        return build_sampler("tabular_sgs", pool, seed=seed, kernel="sharp", history=100)
    if method == "tabular_sgs_default":
        return build_sampler("tabular_sgs", pool, seed=seed, kernel="default", history=100)
    if method == "knn_sgs":
        return build_sampler("knn_sgs", pool, seed=seed, device=device, k=32, tau=50.0, kernel="sharp",
                             history_limit=100_000, max_anchors=16_384, query_chunk_size=8192)
    if method == "nsgs":
        return build_sampler("nsgs", pool, seed=seed, device=device, num_members=5,
                             hidden_dims=(256, 256, 256), optimizer_steps=20, batch_size=4096,
                             cold_updates=20)
    if method == "nsgs_v2":
        v2_kernel = SGSKernel(target=0.66, concentration=5.0, temperature=2.0, epsilon=1e-5, uniform_floor=0.0)
        return build_sampler("nsgs_v2", pool, seed=seed, device=device, kernel=v2_kernel,
                             history_limit=1_000_000, tau=120.0, beta_optimism=0.30, adaptive_target=True,
                             num_members=5, hidden_dims=(256, 256, 256), optimizer_steps=20, batch_size=4096,
                             cold_updates=20)
    if method in ("uniform", "plr", "neural_plr"):
        kw = {"device": device} if method == "neural_plr" else {}
        return build_sampler(method, pool, seed=seed, **kw)
    raise ValueError(f"Unsupported method: {method}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--catalog", type=Path, default=EXP_ROOT / "catalog")
    p.add_argument("--seed", type=int, default=73205)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--updates", type=int, default=5000)
    p.add_argument("--eval-at", type=int, nargs="*", default=[500, 1000, 2000, 3000, 4000, 5000])
    p.add_argument("--sampler-update-every", type=int, default=None,
                   help="Override how often step_update() runs (default: 1 for fast, 5 for knn/nsgs).")
    p.add_argument("--fwd-mode", default=None,
                   help="Override recipe fwd_mode (default: None -> use recipe's own fwd_mode).")
    p.add_argument("--haa-target-map", choices=["none", "soft_tanh"], default="none")
    p.add_argument("--haa-limit-pen-weight", type=float, default=None,
                   help="Override HAA soft-limit penalty weight (default: None -> recipe default or 0.0).")
    p.add_argument("--haa-soft-factor", type=float, default=None,
                   help="Override HAA soft-limit factor (default: None -> recipe default or 0.95).")
    p.add_argument("--recipe", default="recipe-13t-v3")
    p.add_argument("--device-index", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    t_boot = monotonic()

    from isaaclab.app import AppLauncher
    launcher = AppLauncher(headless=True)
    _app = launcher.app

    import torch
    import navigation as nav
    from ppo_smoke import make_algorithm, obs, fingerprint
    from trimesh_scene import TrimeshBatchedScene

    recipe = args.recipe
    rcfg = nav.get_recipe(recipe)
    gamma = float(rcfg.get("gamma", 0.998))
    entropy_coef = float(rcfg.get("entropy_coef", 0.002))
    hygiene = bool(rcfg.get("hygiene", 1))
    fwd_mode = args.fwd_mode if args.fwd_mode is not None else str(rcfg.get("fwd_mode", "first500"))
    haa_pen_w = float(args.haa_limit_pen_weight) if args.haa_limit_pen_weight is not None else float(rcfg.get("default_haa_limit_pen_weight", 0.0))
    haa_soft_fac = float(args.haa_soft_factor) if args.haa_soft_factor is not None else float(rcfg.get("default_haa_soft_factor", 0.95))
    terrain_friction = float(rcfg.get("terrain_friction", 1.0))
    scan_offset_x = float(rcfg.get("scan_offset_x", 0.0))
    scan_size = tuple(float(x) for x in rcfg.get("scan_size", (1.6, 1.0)))
    void_mode = str(rcfg.get("void_mode", "legacy_neg"))
    dist_mode = str(rcfg.get("dist_mode", "gaussian"))
    init_std = float(rcfg.get("init_std", 1.0))
    std_bounds = tuple(float(x) for x in rcfg.get("std_bounds", (0.06, 1.35)))
    use_critic_priv = bool(rcfg.get("use_critic_priv", False))
    use_clipped_vl = rcfg.get("use_clipped_value_loss", None)
    max_episode_steps = 1000

    pool, meta = load_task_pool(args.catalog)
    train_arr = np.load(args.catalog / "train_tasks.npy", allow_pickle=False)
    strict_bytes = (args.catalog / "strict_routes.json").read_bytes()
    strict_sha256 = hashlib.sha256(strict_bytes).hexdigest()
    task_terrain = (train_arr[:, 0].astype(np.int64) // 20)  # (104000,) in 0..12

    world = TrimeshBatchedScene(args.catalog, args.num_envs, terrain_friction=terrain_friction,
                                max_episode_steps=max_episode_steps,
                                haa_target_map=args.haa_target_map,
                                haa_limit_pen_weight=haa_pen_w,
                                haa_soft_factor=haa_soft_fac,
                                scan_offset_x=scan_offset_x,
                                scan_size=scan_size,
                                void_mode=void_mode)
    world.select_bank("train")

    # Deterministic identical initialization across ALL arms!
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sampler = make_sampler(args.method, pool, meta, seed=args.seed, device=world.device)
    # Initial reset batch uses a dedicated seed-identical uniform draw so EVERY arm starts
    # on the EXACT SAME initial task assignment & zero-update weights!
    init_rng = np.random.default_rng(args.seed + 101)
    initial_ids_np = init_rng.integers(0, pool.num_tasks, size=args.num_envs, dtype=np.int64)
    initial_ids = torch.as_tensor(initial_ids_np, device=world.device, dtype=torch.long)
    world.place(world.env_ids, initial_ids)

    readings = obs(world, clip_height=hygiene, include_critic=use_critic_priv)
    alg = make_algorithm(
        world, readings, args.updates,
        gamma=gamma, hygiene=hygiene, entropy_coef=entropy_coef,
        dist_mode=dist_mode, init_std=init_std, std_bounds=std_bounds,
        use_critic_priv=use_critic_priv, use_clipped_value_loss=use_clipped_vl,
    )
    birth_gen = torch.Generator(device=world.device).manual_seed(args.seed + 2)
    world.steps[:] = torch.randint(world.max_episode_steps, (args.num_envs,), device=world.device, generator=birth_gen)

    update_every = args.sampler_update_every
    if update_every is None:
        update_every = 5 if args.method in ("knn_sgs", "nsgs", "nsgs_v2", "neural_plr") else 1

    initial_state = dict(
        method=args.method,
        seed=args.seed,
        num_envs=args.num_envs,
        updates=args.updates,
        steps_per_update=STEPS,
        recipe=recipe,
        gamma=gamma,
        entropy_coef=entropy_coef,
        dist_mode=dist_mode,
        init_std=init_std,
        std_bounds=list(std_bounds),
        use_critic_priv=use_critic_priv,
        use_clipped_value_loss=use_clipped_vl,
        hygiene=hygiene,
        fwd_mode=fwd_mode,
        haa_target_map=args.haa_target_map,
        haa_limit_pen_weight=haa_pen_w,
        haa_soft_factor=haa_soft_fac,
        haa_limits_info=world.haa_limits_info,
        terrain_friction=terrain_friction,
        scan_offset_x=scan_offset_x,
        scan_size=list(scan_size),
        void_mode=void_mode,
        obs_dim=int(readings["policy"].shape[1]),
        critic_obs_dim=int(readings["critic"].shape[1]) if use_critic_priv else int(readings["policy"].shape[1]),
        max_episode_steps=max_episode_steps,
        sampler_update_every=update_every,
        catalog_path=str(args.catalog.resolve()),
        catalog_version=meta.get("version", "unknown"),
        catalog_train_sha256=meta["train_sha256"],
        catalog_holdout_sha256=meta["holdout_sha256"],
        catalog_collision_sha256=meta["collision_sha256"],
        catalog_strict_sha256=strict_sha256,
        maze_geodesic_loaded=bool(world.maze_fields is not None),
        resolved_weights_schedule={
            f"u{it:04d}": [round(float(x), 8) for x in nav.reward_weights(it, fwd_mode=fwd_mode, recipe=recipe)]
            for it in (0, 499, 500, 749, 999, 1000, 2500)
        },
        actor_sha256=fingerprint(alg.actor.state_dict()),
        critic_sha256=fingerprint(alg.critic.state_dict()),
        optimizer_sha256=fingerprint(alg.optimizer.state_dict()),
        initial_tasks_sha256=hashlib.sha256(initial_ids_np.tobytes()).hexdigest(),
        world_steps_sha256=fingerprint(world.steps),
        device_index=args.device_index,
    )
    atomic_json(args.out / "initial.json", initial_state)
    print(f"[INIT] {args.method} actor={initial_state['actor_sha256'][:12]} init_tasks={initial_state['initial_tasks_sha256'][:12]} update_every={update_every}", flush=True)

    def save_ckpt(update_idx: int, outcomes_arr: np.ndarray, controls: int):
        path = args.out / f"checkpoint-{update_idx:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save(dict(
            method=args.method,
            update=update_idx,
            seed=args.seed,
            controls=controls,
            initial=initial_state,
            ppo=alg.save(),
            sampler_state=sampler.state_dict(),
            natural_outcomes=outcomes_arr.copy(),
            learning_rate=alg.learning_rate,
            run_wall_s=monotonic() - t_boot,
        ), tmp)
        tmp.replace(path)

    outcomes = np.zeros(5, dtype=np.int64)
    save_ckpt(0, outcomes, 0)

    win_draws = np.zeros(len(TERRAIN_NAMES), dtype=np.int64)
    win_ended = np.zeros((len(TERRAIN_NAMES), 5), dtype=np.int64)
    cumulative_controls = 0
    eval_set = set(args.eval_at)
    use_plr_scores = args.method in ("plr", "neural_plr")
    ep_td_abs_sum = torch.zeros(args.num_envs, device=world.device, dtype=torch.float32)
    ep_step_cnt = torch.zeros(args.num_envs, device=world.device, dtype=torch.long)

    for update in range(1, args.updates + 1):
        t_u = monotonic()
        reward_sums = torch.zeros(len(nav.TERMS), device=world.device, dtype=torch.float64)
        obj_sum = torch.zeros((), device=world.device, dtype=torch.float64)
        extra_reg_sum = torch.zeros_like(obj_sum)
        air_bonus_sum = torch.zeros_like(obj_sum)
        alive_bonus_sum = torch.zeros_like(obj_sum)
        raw_reg_sum = torch.zeros((), device=world.device, dtype=torch.float64)
        reg_scale_sum = torch.zeros((), device=world.device, dtype=torch.float64)
        haa_exc_sum = torch.zeros((), device=world.device, dtype=torch.float64)
        haa_pen_sum = torch.zeros((), device=world.device, dtype=torch.float64)
        haa_raw_sum = torch.zeros(4, device=world.device, dtype=torch.float64)
        haa_cmd_sum = torch.zeros(4, device=world.device, dtype=torch.float64)
        haa_meas_sum = torch.zeros(4, device=world.device, dtype=torch.float64)
        haa_lo1_sum = torch.zeros(4, device=world.device, dtype=torch.float64)
        haa_hi1_sum = torch.zeros(4, device=world.device, dtype=torch.float64)
        v5_sums = {}
        alg.train_mode()
        ep_recorded = 0
        with torch.no_grad():
            for _step in range(STEPS):
                v_cur = alg.critic(readings).squeeze(-1) if use_plr_scores else None
                action = alg.act(readings)
                labels, sig = world.step(action, training_iteration=update - 1, recipe=recipe, fwd_mode=fwd_mode)
                next_obs = obs(world, clip_height=hygiene, include_critic=use_critic_priv)
                done = labels.ne(0)
                slots = done.nonzero().flatten()
                raw = sig["reward"]
                obj_sum += raw.sum(dtype=torch.float64)
                extra_reg_sum += sig["extra_reg_scaled"].sum(dtype=torch.float64)
                air_bonus_sum += sig["air_bonus"].sum(dtype=torch.float64)
                alive_bonus_sum += sig["alive_bonus"].sum(dtype=torch.float64)
                reward_sums += sig["reward_terms"].sum(dim=0, dtype=torch.float64)
                raw_reg_sum += sig["raw_reg_x"].sum(dtype=torch.float64)
                reg_scale_sum += sig["reg_scale"].sum(dtype=torch.float64)
                haa_exc_sum += sig["haa_limit_excess"].sum(dtype=torch.float64)
                haa_pen_sum += sig["haa_limit_pen"].sum(dtype=torch.float64)
                haa_raw_sum += sig["q_raw_haa"].sum(dim=0, dtype=torch.float64)
                haa_cmd_sum += sig["q_cmd_haa"].sum(dim=0, dtype=torch.float64)
                haa_meas_sum += sig["q_meas_haa"].sum(dim=0, dtype=torch.float64)
                haa_lo1_sum += sig["near_phys_lo_1deg"].sum(dim=0, dtype=torch.float64)
                haa_hi1_sum += sig["near_phys_hi_1deg"].sum(dim=0, dtype=torch.float64)
                if sig.get("v5_diag") is not None:
                    for dk, dv in sig["v5_diag"].items():
                        v5_sums[dk] = v5_sums.get(dk, 0.0) + float(dv.sum(dtype=torch.float64).item())
                reward = raw * 0.02  # hygiene scaling matching v3
                if use_plr_scores:
                    v_nxt_all = alg.critic(next_obs).squeeze(-1)
                    cont_mask = ((labels == 0) | (labels == 4)).float()
                    td_abs = ((reward + alg.gamma * v_nxt_all * cont_mask) - v_cur).abs()
                    ep_td_abs_sum.add_(td_abs)
                    ep_step_cnt.add_(1)
                if slots.numel():
                    bootstrap = alg.gamma * alg.critic(next_obs).squeeze(-1) * (labels == 4)
                    reward = reward + bootstrap
                    comp_ids = world.task_ids[slots].cpu().numpy()
                    nat = labels[slots].cpu().numpy()
                    outcomes += np.bincount(nat, minlength=5)
                    succ = (nat == 1)
                    if use_plr_scores:
                        ep_scores_np = (
                            ep_td_abs_sum[slots] / ep_step_cnt[slots].clamp_min(1).float()
                        ).detach().cpu().numpy().astype(np.float64)
                        sampler.record(comp_ids, succ, update=update, scores=ep_scores_np)
                        ep_td_abs_sum[slots] = 0.0
                        ep_step_cnt[slots] = 0
                    else:
                        sampler.record(comp_ids, succ, update=update)
                    ep_recorded += len(comp_ids)
                    next_ids_np = sampler.sample(len(comp_ids))
                    next_ids = torch.as_tensor(next_ids_np, device=world.device, dtype=torch.long)
                    # Track per-terrain draws & ended outcomes
                    np.add.at(win_draws, task_terrain[next_ids_np], 1)
                    np.add.at(win_ended, (task_terrain[comp_ids], nat), 1)
                    world.place(slots, next_ids)
                    next_obs = obs(world, clip_height=hygiene, include_critic=use_critic_priv)
                alg.process_env_step(next_obs, reward, done, {})
                readings = next_obs
            alg.compute_returns(readings)

        losses = alg.update()
        torch.cuda.synchronize()
        if not all(np.isfinite(float(v)) for v in losses.values()):
            raise RuntimeError(f"Non-finite PPO loss at u{update}: {losses}")

        # Sampler distribution update
        t_s = monotonic()
        if update % update_every == 0:
            sampler.step_update(update)
        sampler_ms = (monotonic() - t_s) * 1000.0
        cumulative_controls += args.num_envs * STEPS

        sums = (reward_sums / (args.num_envs * STEPS)).cpu().numpy()
        reg_mean = float(sums[[2, 3, 4, 5, 7, 8, 9]].sum())
        has_std = hasattr(alg.actor, "distribution") and hasattr(alg.actor.distribution, "std")
        std_t = alg.actor.distribution.std.detach() if has_std else None
        std_val = float(std_t.mean().item()) if std_t is not None else float("nan")
        std_min = float(std_t.min().item()) if std_t is not None else float("nan")
        std_max = float(std_t.max().item()) if std_t is not None else float("nan")
        s_diag = sampler.diagnostics()
        denominator = args.num_envs * STEPS
        extra_mean = extra_reg_sum.item() / denominator
        air_mean = air_bonus_sum.item() / denominator
        alive_mean = alive_bonus_sum.item() / denominator
        env_mean = obj_sum.item() / denominator
        limit_mean = haa_pen_sum.item() / denominator
        row = dict(
            update=update,
            controls=cumulative_controls,
            sec=round(monotonic() - t_u, 3),
            sampler_ms=round(sampler_ms, 2),
            ep_recorded=ep_recorded,
            outcomes=outcomes.tolist(),
            action_std=round(std_val, 4),
            action_std_min=round(std_min, 4),
            action_std_max=round(std_max, 4),
            reg_mean=round(reg_mean, 5),
            reward_ledger_version=3,
            extra_reg_scaled_mean=round(extra_mean, 8),
            feet_air_bonus_mean=round(air_mean, 8),
            alive_bonus_mean=round(alive_mean, 8),
            env_reward_mean=round(env_mean, 8),
            ppo_env_reward_before_bootstrap_mean=round(.02 * env_mean, 8),
            reward_ledger_residual_mean=round(env_mean - (sums.sum() + extra_mean + air_mean + alive_mean + limit_mean), 9),
            raw_reg_mean=round(float(raw_reg_sum.item() / (args.num_envs * STEPS)), 5),
            reg_scale_mean=round(float(reg_scale_sum.item() / (args.num_envs * STEPS)), 4),
            haa_limit_excess_mean=round(float(haa_exc_sum.item() / (args.num_envs * STEPS)), 6),
            haa_limit_pen_mean=round(float(haa_pen_sum.item() / (args.num_envs * STEPS)), 6),
            haa_names=world.haa_names,
            haa_raw_deg=[round(float(v), 2) for v in (haa_raw_sum / (args.num_envs * STEPS)).cpu().numpy() * (180.0 / math.pi)],
            haa_cmd_deg=[round(float(v), 2) for v in (haa_cmd_sum / (args.num_envs * STEPS)).cpu().numpy() * (180.0 / math.pi)],
            haa_meas_deg=[round(float(v), 2) for v in (haa_meas_sum / (args.num_envs * STEPS)).cpu().numpy() * (180.0 / math.pi)],
            haa_lo1_pct=[round(float(v), 2) for v in (haa_lo1_sum / (args.num_envs * STEPS)).cpu().numpy() * 100.0],
            haa_hi1_pct=[round(float(v), 2) for v in (haa_hi1_sum / (args.num_envs * STEPS)).cpu().numpy() * 100.0],
            term_means=[round(float(x), 5) for x in sums],
            v5_diag_means=({dk: round(dv / denominator, 6) for dk, dv in v5_sums.items()} if v5_sums else None),
            losses={k: round(float(v), 5) for k, v in losses.items()},
            sampler_diag={k: (round(float(v), 5) if isinstance(v, (int, float, np.floating)) and np.isfinite(float(v)) else v)
                          for k, v in s_diag.items() if not isinstance(v, (list, dict))},
        )
        append_jsonl(args.out / "updates.jsonl", row)

        if update % 25 == 0 or update in eval_set or update == args.updates:
            tot_end = int(win_ended.sum())
            tot_suc = int(win_ended[:, 1].sum())
            per_ter = {
                TERRAIN_NAMES[i]: dict(
                    draws=int(win_draws[i]),
                    ended=int(win_ended[i].sum()),
                    succ=int(win_ended[i, 1]),
                    rate=(round(float(win_ended[i, 1] / win_ended[i].sum()), 4) if win_ended[i].sum() else None),
                )
                for i in range(len(TERRAIN_NAMES))
            }
            wrow = dict(update=update, ended=tot_end, succ=tot_suc,
                        rate=(round(tot_suc / tot_end, 4) if tot_end else 0.0),
                        per_terrain=per_ter)
            append_jsonl(args.out / "windows.jsonl", wrow)
            print(f"u{update:04d} [{args.method:11s}] {row['sec']:.2f}s (samp={sampler_ms:.1f}ms) std={std_val:.3f} reg={reg_mean:+.4f} win_succ={tot_suc}/{tot_end} ({100*wrow['rate']:.1f}%) tot_succ={outcomes[1]}", flush=True)
            win_draws[:] = 0
            win_ended[:] = 0

        if update % 250 == 0 or update in eval_set or update == args.updates:
            save_ckpt(update, outcomes, cumulative_controls)

    atomic_json(args.out / "result.json", dict(
        method=args.method, seed=args.seed, num_envs=args.num_envs,
        completed_updates=args.updates, controls=cumulative_controls,
        natural_outcomes=outcomes.tolist(), wall_hours=round((monotonic() - t_boot) / 3600.0, 3),
    ))
    print(f"[DONE] {args.method} completed {args.updates} updates in {(monotonic()-t_boot)/3600:.2f}h", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
