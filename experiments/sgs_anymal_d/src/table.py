import numpy as np
from course import TASK_COUNT

class Table:
    def __init__(self,n=TASK_COUNT):
        self.history=np.zeros((n,100),np.uint8)
        self.seen=np.zeros(n,np.int64)
        self.total=np.zeros(n,np.int64)
        self.success=np.zeros(n,np.int64)
        self.mass=np.full(n,self.weights(np.array([.5]))[0],dtype=np.float64)
    @staticmethod
    def weights(p):
        e=1e-4
        return np.sqrt(np.maximum((p+e)**3.3*(1-p+e)**1.7,e)+e)
    def record(self,ids,ys):
        ids,ys=np.asarray(ids,np.int64),np.asarray(ys,np.uint8)
        if not ids.size:return
        perm=np.argsort(ids,kind='stable'); keys=ids[perm]; labels=ys[perm]
        changed,starts,counts=np.unique(keys,return_index=True,return_counts=True)
        position=np.arange(len(keys))-np.repeat(starts,counts)
        keep=position>=np.repeat(np.maximum(counts-100,0),counts)
        self.history[keys[keep],(self.seen[keys[keep]]+position[keep])%100]=labels[keep]
        self.seen[changed]+=counts
        self.success[changed]+=np.add.reduceat(labels,starts,dtype=np.int64)
        self.total[changed]=self.history[changed].sum(axis=1)
        self.mass[changed]=self.weights(self.total[changed]/np.minimum(self.seen[changed],100))
    def state(self):return vars(self)
    def load(self,state):
        for key in vars(self):getattr(self,key)[:]=state[key]

