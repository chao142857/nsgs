"""Online neural probability estimates using the frozen K2 sampling kernels."""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
import time

import numpy as np
import torch
from torch import nn

from .sgs_kernels import EMPTY_HISTORY_PRIOR, KERNELS

MEMBERS = 5
COLD_UPDATES = 20
HISTORY_LIMIT = 1_000_000
OPTIMIZER_STEPS = 20
BATCH_SIZE = 4096
FEATURE_NAMES = ('target_x','target_y','target_z','horizontal_distance','azimuth_sin',
                 'azimuth_cos','height','reference_ee_distance','episode_seconds')


@dataclass(frozen=True)
class EngineeringOptions:
    kernel: str
    tau: int
    alpha: int | None
    features: str = 'raw'
    feature_contract: str = 'engineering'
    adam_epsilon: float = 1e-8
    learning_rate: float = 3e-4

    def __post_init__(self):
        if self.kernel not in KERNELS or self.tau not in (50,100,200,400):
            raise ValueError('Expected a K2 kernel and a planned decay time')
        if self.alpha not in (None,1,2,5,10) or self.features not in ('raw','fourier','drop_distances','drop_azimuth'):
            raise ValueError('Expected a planned feature or mixing candidate')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Learning rate must be positive')
        if self.feature_contract not in ('engineering','p2c_G5') or not 0 < self.adam_epsilon < 1:
            raise ValueError('Invalid feature or optimizer contract')


def feature_rows(pool, variant, reference_ee):
    target = np.asarray(pool['target'],dtype=np.float32)
    distance = np.linalg.norm(target[:,:2],axis=1)
    azimuth = np.asarray(pool['azimuth'],dtype=np.float32)
    reference = np.asarray(reference_ee,dtype=np.float32)
    if reference.shape != (3,):
        raise ValueError('Expected fixed reference end effector xyz')
    toward_reference = np.linalg.norm(target-reference,axis=1)
    raw = np.column_stack((target,distance,np.sin(azimuth),np.cos(azimuth),
                           pool['height'],toward_reference,pool['episode_seconds']))
    # Engineering candidate only. A 2C decision must bind final channels and scales.
    scales = np.array([3.,3.,1.,3.,1.,1.,1.,3.,10.],np.float32)
    rows = raw / scales
    if variant == 'fourier':
        angles = rows[:,None,:] * np.array([1.,2.,4.],np.float32)[None,:,None] * np.pi
        rows = np.column_stack((rows,np.sin(angles).reshape(len(rows),-1),np.cos(angles).reshape(len(rows),-1)))
    if not np.isfinite(rows).all():
        raise ValueError('Non-finite configuration features')
    return np.ascontiguousarray(rows,dtype=np.float32)


def p3b_features(pool, variant):
    # The frozen 2C G5 transform uses only fixed point geometry.
    a = pool['azimuth']
    raw = np.column_stack((pool['target'],pool['horizontal_distance'],np.sin(a),np.cos(a),
                           pool['nominal_distance'],pool['episode_seconds'])).astype(np.float64)
    std = raw.std(axis=0)
    valid = std>1e-12
    mean=raw.mean(axis=0)
    raw = (raw-mean)/np.where(valid,std,1)
    if variant=='fourier':
        phase = np.pi*raw[:,valid]
        rows = np.column_stack((raw,np.sin(phase),np.cos(phase),np.sin(2*phase),np.cos(2*phase)))
    elif variant=='drop_distances':
        rows = raw[:,[0,1,2,4,5,7]]
    elif variant=='drop_azimuth':
        rows = raw[:,[0,1,2,3,6,7]]
    else:
        rows = raw
    return np.ascontiguousarray(rows,dtype=np.float32), dict(mean=mean.tolist(),std=std.tolist(),
        nonconstant=valid.tolist(),output_columns=rows.shape[1])

def kernel_probabilities(rates, kernel):
    """Retain the K2 table path's float64 floor, epsilon, log and temperature order."""
    config = KERNELS[kernel]
    epsilon, target, concentration = (config[key] for key in ('epsilon','target','concentration'))
    p = np.asarray(rates,dtype=np.float64)
    if p.ndim!=1 or not p.size or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected one finite probability in [0,1] per task')
    weight = (p+epsilon)**(concentration*target) * (1-p+epsilon)**(concentration*(1-target))
    logits = np.log(np.maximum(weight,epsilon)+epsilon)/config['temperature']
    w = np.exp(logits-logits.max())
    return w/w.sum()


class Replay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.task = np.empty(capacity,np.int32)
        self.end_update = np.empty(capacity,np.int32)
        self.success = np.empty(capacity,np.uint8)
        self.size = self.cursor = 0

    def append(self, task, updates, success):
        if len(task)>self.capacity:
            tail = slice(-self.capacity,None)
            task,updates,success = task[tail],updates[tail],success[tail]
        idx = (np.arange(len(task))+self.cursor)%self.capacity
        self.task[idx],self.end_update[idx],self.success[idx] = task,updates,success
        self.size = min(self.capacity,self.size+len(task))
        self.cursor = (self.cursor+len(task))%self.capacity

    def state_dict(self):
        return dict(capacity=self.capacity,cursor=self.cursor,size=self.size,
                    task=self.task[:self.size].copy(),end_update=self.end_update[:self.size].copy(),
                    success=self.success[:self.size].copy())

    def load_state_dict(self, state):
        size,cursor = state['size'],state['cursor']
        if state['capacity']!=self.capacity or not 0<=size<=self.capacity or not 0<=cursor<self.capacity:
            raise ValueError('Replay capacity or cursor differs')
        if cursor!=size and size!=self.capacity:
            raise ValueError('Non-full replay cursor differs')
        for name in ('task','end_update','success'):
            dest=getattr(self,name)
            source=state[name]
            if source.shape!=dest[:size].shape or source.dtype!=dest.dtype:
                raise ValueError('Replay checkpoint arrays differ')
            dest[:size]=source
        self.size,self.cursor = size,cursor


class NeuralSGS:
    def __init__(self, features, options, seed, device='cpu', *,
                 capacity=HISTORY_LIMIT, batch_size=BATCH_SIZE):
        matrix = np.ascontiguousarray(features,dtype=np.float32)
        if matrix.ndim!=2 or not matrix.size or not np.isfinite(matrix).all() or capacity<1 or batch_size<1:
            raise ValueError('Invalid immutable scorer features or buffer capacity')
        self.options,self.batch_size = options,batch_size
        self.features_sha256 = hashlib.sha256(matrix.tobytes()).hexdigest()
        self.device = torch.device(device)
        with torch.inference_mode(False):
            self.features = torch.as_tensor(matrix.copy(),device=self.device)
        self.replay = Replay(capacity)
        children = np.random.SeedSequence(seed).spawn(2*MEMBERS)
        self.batch_rng = [np.random.default_rng(s) for s in children[MEMBERS:]]
        self.models=[]
        # Only seed the saved CPU generator. Torch manual_seed also touches CUDA.
        with torch.inference_mode(False), torch.random.fork_rng(devices=[]):
            for child in children[:MEMBERS]:
                torch.random.default_generator.manual_seed(int(child.generate_state(1)[0]))
                self.models.append(nn.Sequential(nn.Linear(matrix.shape[1],256),nn.ReLU(),nn.Linear(256,256),
                    nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,1)).to(self.device))
        self.optimizers=[torch.optim.Adam(model.parameters(),lr=options.learning_rate,eps=options.adam_epsilon) for model in self.models]
        self.completed = np.zeros(len(matrix),np.int64)
        self.successful = np.zeros(len(matrix),np.int64)
        self.update=0
        self.fit_steps=0
        self.neural_rate=np.full(len(matrix),EMPTY_HISTORY_PRIOR)
        self.member_disagreement=np.zeros(len(matrix),np.float64)
        self.probabilities=kernel_probabilities(self.neural_rate,options.kernel)

    def advance(self, update, task_ids, end_updates, successes, timings=None):
        if type(update) is not int or update!=self.update+1:
            raise ValueError('Expected the next completed PPO update')
        tasks = np.asarray(task_ids,dtype=np.int64)
        ends = np.asarray(end_updates,dtype=np.int64)
        labels = np.asarray(successes)
        if tasks.ndim!=1 or ends.shape!=tasks.shape or labels.shape!=tasks.shape or np.any(ends!=update):
            raise ValueError('Only natural training outcomes from this completed update may enter online replay')
        if np.any((tasks<0)|(tasks>=len(self.completed))) or np.any((labels!=0)&(labels!=1)):
            raise ValueError('Out-of-range task or non-binary outcome')
        self.update=update
        if len(tasks):
            np.add.at(self.completed,tasks,1)
            np.add.at(self.successful,tasks,labels.astype(np.int64))
            self.replay.append(tasks,ends,labels.astype(np.uint8))
        if self.replay.size:
            if timings is not None:
                if self.device.type=='cuda':
                    torch.cuda.synchronize(self.device)
                start=time.monotonic()
            self.fit()
            if timings is not None:
                if self.device.type=='cuda':
                    torch.cuda.synchronize(self.device)
                timings['network_fit_s']=time.monotonic()-start
            if update>=COLD_UPDATES:
                if timings is not None:
                    start=time.monotonic()
                self.predict()
                if timings is not None:
                    if self.device.type=='cuda':
                        torch.cuda.synchronize(self.device)
                    timings['full_pool_inference_s']=time.monotonic()-start
        return self.probabilities.copy()

    def fit(self):
        with torch.inference_mode(False), torch.enable_grad():
            for model,optimizer,rng in zip(self.models,self.optimizers,self.batch_rng):
                model.train()
                for _ in range(OPTIMIZER_STEPS):
                    index=rng.integers(0,self.replay.size,size=self.batch_size)
                    # Each ensemble member has its own bootstrap stream, with
                    # replacement from only the completed training buffer.
                    weights=np.exp(-(self.update-self.replay.end_update[index])/self.options.tau)
                    t=torch.as_tensor(self.replay.task[index],device=self.device,dtype=torch.int64)
                    y=torch.as_tensor(self.replay.success[index],device=self.device,dtype=torch.float32)
                    w=torch.as_tensor(weights/weights.sum(),device=self.device,dtype=torch.float32)
                    optimizer.zero_grad(set_to_none=True)
                    logits=model(self.features[t]).squeeze(-1)
                    loss=(nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum()
                    if not torch.isfinite(loss).item():
                        raise ValueError('Non-finite scorer loss')
                    loss.backward()
                    optimizer.step()
                    self.fit_steps+=1

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
        self.neural_rate=members.mean(axis=0)
        self.member_disagreement=members.std(axis=0)
        alpha=self.options.alpha
        estimate=self.neural_rate if alpha is None else (alpha*self.neural_rate+self.successful)/(alpha+self.completed)
        self.probabilities=kernel_probabilities(estimate,self.options.kernel)

    def _checkpoint_options(self):
        options = asdict(self.options)
        if self.options.feature_contract=='engineering' and self.options.adam_epsilon==1e-8:
            del options['feature_contract'], options['adam_epsilon']
        return options

    def state_dict(self):
        return dict(version=1,options=self._checkpoint_options(),features_sha256=self.features_sha256,
            batch_size=self.batch_size,update=self.update,fit_steps=self.fit_steps,replay=self.replay.state_dict(),
            batch_rng=[deepcopy(rng.bit_generator.state) for rng in self.batch_rng],
            models=[deepcopy(model.state_dict()) for model in self.models],
            optimizers=[deepcopy(opt.state_dict()) for opt in self.optimizers],
            completed=self.completed.copy(),successful=self.successful.copy(),neural_rate=self.neural_rate.copy(),
            member_disagreement=self.member_disagreement.copy(),probabilities=self.probabilities.copy())

    def load_state_dict(self, state):
        if (state['version'],state['options'],state['features_sha256'],state['batch_size'])!=(
                1,self._checkpoint_options(),self.features_sha256,self.batch_size):
            raise ValueError('Neural SGS checkpoint configuration or feature hash differs')
        for current,payload in zip(self.models,state['models'],strict=True):
            current.load_state_dict(payload)
        for current,payload in zip(self.optimizers,state['optimizers'],strict=True):
            current.load_state_dict(payload)
        for current,payload in zip(self.batch_rng,state['batch_rng'],strict=True):
            current.bit_generator.state=deepcopy(payload)
        self.replay.load_state_dict(state['replay'])
        self.update,self.fit_steps=state['update'],state['fit_steps']
        for key in ('completed','successful','neural_rate','member_disagreement','probabilities'):
            value=np.asarray(state[key])
            if value.shape!=getattr(self,key).shape or not np.isfinite(value).all():
                raise ValueError('Neural SGS checkpoint array differs: '+key)
            setattr(self,key,value.copy())
        if np.any(self.successful>self.completed) or np.any(self.replay.end_update[:self.replay.size]>self.update):
            raise ValueError('Impossible completed or future results in scorer checkpoint')
