"""Assign the immutable terrain and the original birth/goal in one native reset."""
from pathlib import Path

import numpy as np
import torch
from goalreach import mdp
from goalreach.core import append_rows, write_json

from samplers import GRID, METHODS, sampler_for
from planes import PersistentPlanes, configure_planes, orient_births


NativeRuntime = mdp.TaskRuntime


class SpatialRuntime(NativeRuntime):
    def __init__(self, env):
        super().__init__(env)
        settings = env.cfg.goalreach
        assert not settings.get('p3b')
        self.method = settings['spatial_method']
        if self.method != 'evaluation':
            self.p3b = True
            self.sampler = sampler_for(self.method, self.pool, self.sampler_seed)
            self.grid_bins = GRID.get(self.method, (None, None))[1]
            if self.method == 'nsgs_e4_sharp':
                from spatial_scorer import TerrainNeuralTraining
                learner = settings.get('learner_device',env.device)
                self.neural = TerrainNeuralTraining(self.sampler,self.pool,self.sampler_seed,learner)
                write_json(self.out/'spatial_features.json', self.neural.feature_metadata)
        if not hasattr(env, 'episode_planes'):
            env.episode_planes = PersistentPlanes(env)
        self.planes = env.episode_planes
        orient_births(self)
        self.discarded = 0
        self.eval_repeat = None

    def assign(self, ids):
        if self.method == 'evaluation' and self.forced_ids is None and self.eval_repeat is not None:
            self.forced_ids = self.eval_repeat[ids].cpu().tolist()
        tasks = super().assign(ids)
        self.planes.assign(ids,self.mu[tasks],self.alpha[tasks],self.plane_rotations[tasks])
        return tasks

    @torch.inference_mode()
    def discard_at_evaluation_boundary(self):
        self.discarded += int((self.state.task >= 0).sum())
        self.state.reason.zero_()
        self.state.task.fill_(-1)

    def finish(self, ids):
        done = ids[self.state.reason[ids] != 0]
        before = len(self.last_rows)
        if len(done):
            tasks = self.state.task[done].cpu().numpy()
            for slot, row in zip(done.cpu().tolist(), tasks):
                expected = (float(self.pool['mu'][row]),float(self.pool['alpha_deg'][row]))
                assert self.planes.keys[slot] == expected
        super().finish(ids)
        for row in self.last_rows[before:]:
            task = row['task_id']
            row.update(target_index=int(self.pool['target_index'][task]),
                       config_id=int(self.pool['config_id'][task]),
                       mu=float(self.pool['mu'][task]),alpha_deg=float(self.pool['alpha_deg'][task]),
                       cell_id=int(self.pool['cell_id'][task]))

    def end_update(self, update):
        from goalreach.neural_sgs import COLD_UPDATES, kernel_probabilities

        if self.neural:
            model = self.neural.estimator
            timings = dict(network_fit_s=0., full_pool_inference_s=0.)
            labels, steps = len(self.neural.pending), model.fit_steps
            stage = 'scorer_step'
            target = None
            draw_target = None
            try:
                self.neural.end_update(update, timings)
                stage = 'cold_start'
                if update < COLD_UPDATES:
                    assert np.all(model.neural_rate == .5)
                stage = 'scorer_optimizer_count'
                if model.replay.size:
                    assert model.fit_steps-steps == 100
                if update <= 2 or update in (20,21,101) or update % 100 == 0 or steps == 0:
                    stage = 'probability_guard'
                    weights = self.sampler.probabilities()
                    target = kernel_probabilities(model.neural_rate, 'sharp')
                    # Compare in the sampler's probability coordinates. Its
                    # external vector and draw cache each normalize once.
                    external_target = target/target.sum()
                    draw_target = external_target/external_target.sum()
                    stage = 'probability_kernel_output'
                    assert np.allclose(model.probabilities,target,rtol=1e-14,atol=0.), 'pure sharp kernel output differs'
                    stage = 'probability_external'
                    assert np.allclose(self.sampler.external,external_target,rtol=1e-14,atol=0.), 'external normalization differs'
                    stage = 'probability_draw'
                    assert np.allclose(weights,draw_target,rtol=1e-14,atol=0.), 'draw normalization differs'
                    append_rows(self.out/'spatial_scorer.jsonl', [dict(update=update, natural_labels=labels,
                        natural_labels_total=int(model.completed.sum()), optimizer_steps=model.fit_steps, probability_pure=True,
                        probability_checks=dict(kernel_output=True,external_normalization=True,draw_normalization=True),
                        probability_range=[float(weights.min()), float(weights.max())], **timings)])
            except (AssertionError, ValueError, FloatingPointError) as error:
                from probability_evidence import record_fault

                self.scorer_fault_update = update
                try:
                    record_fault(self.out, update, stage, error, model, self.sampler, target,
                                 kernel_probabilities=kernel_probabilities, rtol=1e-14,
                                 guard_target=draw_target)
                except Exception as evidence_error:
                    error.add_note(f'Probability evidence could not be fully saved: {evidence_error!r}')
                raise
            finally:
                model.diagnostic_members = None
                model.diagnostic_prediction_update = None
        if update <= 2 or update == 101 or update % 100 == 0:
            from grid_formal_protocol import ARM as SELECTED_GRID
            if self.method == SELECTED_GRID:
                from formal_grid4096_runtime import committed
                committed(self,update)
            append_rows(self.out/'spatial_training.jsonl', [dict(update=update,
                drawn=int(self.sampler.sample_counts.sum()), natural_ends=int(self.sampler.completed_counts.sum()),
                wins=int(self.sampler.success_counts.sum()), discarded=self.discarded, planes=self.planes.counters.copy(),
                grid=self.sampler.diagnostics() if self.grid_bins else None)])
        if self.episode_log:
            self.episode_log.flush(update)


def configure(cfg, *, scene_queries=True, probe=False):
    configure_planes(cfg,scene_queries=scene_queries,probe=probe)
    mdp.TaskRuntime = SpatialRuntime
