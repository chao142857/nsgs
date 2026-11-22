import numpy as np
def positions():
    index=np.arange(32768)
    within=index%8192
    return np.stack(((45-within//91)*11.,(within%91-45)*11.,(index//8192-1.5)*32.),axis=1)
def install(out):
    from isaacsim.core.cloner import GridCloner
    from prepare import put
    origins=positions()
    assert len(np.unique(origins,axis=0))==32768
    assert np.max(np.abs(origins[:,:2]))==495 and np.min(np.diff(np.unique(origins[:,2])))==32
    original=GridCloner.get_clone_transforms
    def placement(self,num_clones,position_offsets=None,orientation_offsets=None):
        if num_clones!=32768:
            return original(self,num_clones,position_offsets,orientation_offsets)
        if position_offsets is not None or orientation_offsets is not None or self._spacing!=11:
            raise ValueError('Unsupported custom offset or frozen inter-slot separation')
        self._positions=origins.tolist()
        self._orientations=[[1.,0.,0.,0.] for i in range(num_clones)]
        return self._positions,self._orientations
    GridCloner.get_clone_transforms=placement
    put(out/'origin-layout.json',dict(positions='4 translated layers of 8192',horizontal_spacing_m=11,vertical_spacing_m=32,platform_m=[9.5,9.5,.4],z_levels=sorted(set(origins[:,2])),max_abs_xy_m=495,root_tolerance_m=.00005,translation_only=True))
def check_env(env,out):
    import torch
    from prepare import put
    from goalreach import mdp
    origin=env.scene.env_origins
    assert torch.equal(origin,torch.as_tensor(positions(),device=env.device,dtype=origin.dtype))
    check=env.scene['robot'].data
    maximum=torch.zeros((),device=env.device)
    count=0
    prior=env.step
    def stepped(actions):
        nonlocal maximum,count
        result=prior(actions)
        height=(check.body_pos_w[:,:,2]-origin[:,2,None]).abs().max()
        maximum=torch.maximum(maximum,height)
        count+=1
        if count%128==0:
            v=float(maximum)
            assert v<12.0,(count,v)
            put(out/'layer-clearance.json',dict(controls=count,maximum_robot_link_vertical_offset_m=v,nearest_other_layer_m=32,maximum_rotated_plane_vertical_extent_m=2.0))
        return result
    env.step=stepped

