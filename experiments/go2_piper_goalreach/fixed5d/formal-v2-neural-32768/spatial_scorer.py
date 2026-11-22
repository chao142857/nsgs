"""Original E4 fitting and sharp kernel using only the five shared coordinates."""
from hashlib import sha256

import numpy as np
import torch
from goalreach.neural_runtime import NeuralTraining
from goalreach.neural_sgs import BATCH_SIZE, EngineeringOptions, NeuralSGS, kernel_probabilities


SETTINGS = dict(kernel='sharp', tau=100, alpha=None, features='fourier',
                feature_contract='p2c_G5', learning_rate=.0003, adam_epsilon=1e-8)
FIELDS = ('horizontal_distance','height','mu','alpha_deg')
ENCODED = ('horizontal_distance','azimuth_sin','azimuth_cos','height','mu','alpha_deg')


def features(pool):
    theta = np.asarray(pool['azimuth'], dtype=np.float64)
    radius, height, mu, alpha = [np.asarray(pool[key], dtype=np.float64) for key in FIELDS]
    raw = np.column_stack((radius,np.sin(theta),np.cos(theta),height,mu,alpha))
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    valid = std > 1.e-12
    raw = (raw-mean)/np.where(valid,std,1.)
    phase = np.pi*raw[:,valid]
    matrix = np.ascontiguousarray(np.column_stack((raw,np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase))),
                                  dtype='<f4')
    if not np.all(np.isfinite(matrix)):
        raise ValueError('Nonfinite terrain descriptors')
    metadata = dict(coordinates=ENCODED, mean=mean.tolist(), std=std.tolist(), nonconstant=valid.tolist(),
                    shape=list(matrix.shape), sha256=sha256(matrix.tobytes()).hexdigest())
    return matrix, metadata


class DiagnosticNeuralSGS(NeuralSGS):
    """Keep the original predictor's actual member array until the audit finishes."""

    diagnostic_members = None
    diagnostic_prediction_update = None

    @torch.no_grad()
    def predict(self):
        if self.options.feature_contract == 'p2c_G5':
            prediction = torch.empty((len(self.models),len(self.features)),device=self.device)
            for model, destination in zip(self.models,prediction):
                model.eval()
                for chunk, output in zip(self.features.split(BATCH_SIZE),destination.split(BATCH_SIZE)):
                    output.copy_(model(chunk).squeeze(-1).sigmoid())
            members = prediction.cpu().numpy().astype(np.float64)
        else:
            predictions=[]
            for model in self.models:
                model.eval()
                probabilities=torch.cat([model(chunk).squeeze(-1).sigmoid().cpu()
                                         for chunk in self.features.split(BATCH_SIZE)])
                predictions.append(probabilities.numpy().astype(np.float64))
            members=np.asarray(predictions)
        self.diagnostic_members = members
        self.diagnostic_prediction_update = self.update
        self.neural_rate=members.mean(axis=0)
        self.member_disagreement=members.std(axis=0)
        alpha=self.options.alpha
        estimate=self.neural_rate if alpha is None else (alpha*self.neural_rate+self.successful)/(alpha+self.completed)
        self.probabilities=kernel_probabilities(estimate,self.options.kernel)


class TerrainNeuralTraining(NeuralTraining):
    def __init__(self, sampler, pool, seed, device):
        matrix, self.feature_metadata = features(pool)
        self.sampler = sampler
        self.pending = []
        self.estimator = DiagnosticNeuralSGS(matrix, EngineeringOptions(**SETTINGS), seed, device)
        self.sampler.set_probabilities(self.estimator.probabilities)
