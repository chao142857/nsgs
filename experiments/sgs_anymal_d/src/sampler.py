import numpy as np
import torch
from table import Table


class Sampler:
    def __init__(self,seed,device):
        self.device=device
        self.rng=torch.Generator(device=device).manual_seed(seed+1)
        self.table=Table()
        self.refresh()

    def refresh(self):
        self.cdf=torch.as_tensor(self.table.mass,dtype=torch.float64,device=self.device).cumsum(0)

    def record(self,task,outcomes):
        self.table.record(task,outcomes)
        self.refresh()

    def sample(self,count):
        u=torch.rand(count,device=self.device,dtype=torch.float64,generator=self.rng)*self.cdf[-1]
        return torch.searchsorted(self.cdf,u,right=True).clamp_max_(len(self.cdf)-1)
