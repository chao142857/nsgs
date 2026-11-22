"""Bind the online scorer to training-only natural endings and external sampling."""
import hashlib

import numpy as np
import torch

from .neural_sgs import EngineeringOptions, NeuralSGS, feature_rows, p3b_features


class NeuralTraining:
    def __init__(self, sampler, pool, reference_ee, settings, seed, device, **test_limits):
        self.sampler = sampler
        if sampler.interface_mode != 'external':
            raise ValueError('Neural SGS needs the training external-probability sampler')
        options = EngineeringOptions(**settings)
        self.feature_metadata = None
        if options.feature_contract == 'p2c_G5':
            features, self.feature_metadata = p3b_features(pool,options.features)
        else:
            features = feature_rows(pool,options.features,reference_ee)
        self.estimator = NeuralSGS(features,options,seed,device,**test_limits)
        sampler.set_probabilities(self.estimator.probabilities)
        self.pending = []

    def record(self, rows):
        self.pending.extend((row['task_id'],row['end_update'],row['success']) for row in rows)

    def end_update(self, update, timings=None):
        payload = np.asarray(self.pending,dtype=np.int64).reshape(-1,3)
        probabilities = self.estimator.advance(update,payload[:,0],payload[:,1],payload[:,2],timings)
        self.pending.clear()
        self.sampler.set_probabilities(probabilities)

    def state_dict(self):
        return dict(estimator=self.estimator.state_dict(),pending=list(self.pending))

    def load_state_dict(self, state):
        self.estimator.load_state_dict(state['estimator'])
        self.pending = list(state['pending'])
        self.sampler.set_probabilities(self.estimator.probabilities)

    def digest(self):
        """Compare an evaluation boundary without touching scorer or global RNG."""
        checksum = hashlib.sha256()
        def feed(value):
            if isinstance(value, torch.Tensor):
                feed(value.detach().cpu().contiguous().numpy())
            elif isinstance(value,np.ndarray):
                checksum.update(repr((value.dtype.str,value.shape)).encode())
                checksum.update(value.tobytes())
            elif isinstance(value,dict):
                for key,item in value.items():
                    feed(key)
                    feed(item)
            elif isinstance(value,(tuple,list)):
                for item in value:
                    feed(item)
            else:
                checksum.update(repr(value).encode())
        feed(self.state_dict())
        return checksum.hexdigest()
