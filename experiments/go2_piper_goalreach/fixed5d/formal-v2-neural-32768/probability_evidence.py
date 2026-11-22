"""Persist the probability fault as it occurred, before any attempt to restore state."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


SCHEMA = 'spatial-probability-fault-v1'


def publish_json(path, data):
    pending = path.with_suffix('.pending')
    with pending.open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
    pending.replace(path)


def finite_number(value):
    value = float(value)
    if np.isfinite(value):
        return value
    return 'NaN' if np.isnan(value) else ('+Inf' if value > 0 else '-Inf')


def describe(values):
    finite = np.isfinite(values)
    return dict(dtype=values.dtype.str, shape=list(values.shape),
                nonfinite=int(values.size-np.count_nonzero(finite)),
                nan=int(np.count_nonzero(np.isnan(values))),
                positive_inf=int(np.count_nonzero(np.isposinf(values))),
                negative_inf=int(np.count_nonzero(np.isneginf(values))))


def compare(actual, expected, *, rtol):
    if actual is None or expected is None or actual.ndim != 1 or actual.shape != expected.shape:
        return None
    with np.errstate(all='ignore'):
        bad = np.flatnonzero(~np.isclose(actual, expected, rtol=rtol, atol=0.))
        both_finite = np.isfinite(actual) & np.isfinite(expected)
        error = np.abs(actual[both_finite]-expected[both_finite])
        relative = np.divide(error, np.abs(expected[both_finite]),
                             out=np.full_like(error,np.inf), where=expected[both_finite]!=0)
        relative[error==0] = 0
    first = int(bad[0]) if bad.size else None
    return dict(differing_rows=int(bad.size), first_row=first,
                first_actual=None if first is None else finite_number(actual[first]),
                first_expected=None if first is None else finite_number(expected[first]),
                max_finite_pair_absolute_error=finite_number(error.max()) if error.size else None,
                max_finite_pair_relative_error=finite_number(relative.max()) if relative.size else None,
                rtol=rtol, atol=0.)


def record_fault(out, update, stage, error, model, sampler, target, *, kernel_probabilities, rtol, guard_target=None):
    out = Path(out)
    observed = dict(member_predictions=model.diagnostic_members, neural_rate=model.neural_rate,
                    estimator_probabilities=model.probabilities, sampler_external=sampler.external,
                    sampler_draw=sampler._probability, sampler_cumulative=sampler._cumulative,
                    independent_kernel=target, aligned_guard_reference=guard_target)
    recomputation = 'original audit call'
    if target is None:
        try:
            observed['independent_kernel'] = kernel_probabilities(model.neural_rate, 'sharp')
            recomputation = 'after original failure'
        except (ValueError, FloatingPointError) as exc:
            recomputation = f'independent kernel unavailable: {exc!r}'
    with np.errstate(all='ignore'):
        kernel = observed['independent_kernel']
        reference_external = None if kernel is None else kernel/kernel.sum()
        probabilities = model.probabilities
        first = probabilities/probabilities.sum()
        second = first/first.sum()
        from_external = None if sampler.external is None else sampler.external/sampler.external.sum()
    observed.update(reference_external=reference_external, estimator_normalized_once=first, estimator_normalized_twice=second,
                    external_normalized_again=from_external)
    arrays = {name: np.asarray(values) for name, values in observed.items() if values is not None}
    name = f'probability_fault_{update:04d}'
    path = out/f'{name}.npz'
    pending = path.with_suffix('.pending')
    with pending.open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    pending.replace(path)
    checksum = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1<<20), b''):
            checksum.update(chunk)
    report = dict(schema=SCHEMA, update=update, phase=stage, error=repr(error),
                  versions=dict(estimator_state=1,sampler_state=1,numpy=np.__version__),
                  scorer_update=model.update, scorer_optimizer_steps=model.fit_steps,
                  kernel=model.options.kernel, alpha=model.options.alpha, replay_size=model.replay.size,
                  prediction_update=model.diagnostic_prediction_update,
                  independent_kernel_source=recomputation, archive=path.name, archive_sha256=checksum.hexdigest(),
                  arrays={name: describe(values) for name, values in arrays.items()},
                  unavailable=[name for name, values in observed.items() if values is None],
                  comparisons=dict(
                      published_guard=compare(sampler._probability,
                          guard_target if guard_target is not None else observed['independent_kernel'],rtol=rtol),
                      draw_against_raw_kernel=compare(sampler._probability,observed['independent_kernel'],rtol=rtol),
                      estimator_against_independent=compare(probabilities,observed['independent_kernel'],rtol=rtol),
                      external_against_aligned_kernel=compare(sampler.external,reference_external,rtol=rtol),
                      sampler_first_normalization=compare(sampler.external,first,rtol=0.),
                      sampler_second_normalization=compare(sampler._probability,from_external,rtol=0.),
                      draw_against_estimator=compare(sampler._probability,second,rtol=0.)),
                  snapshot_purpose='diagnostic; resumption has not been validated')
    publish_json(out/f'{name}.json', report)
