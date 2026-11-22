"""Bounded-memory statistics for completed training episodes only."""
from collections import Counter


class TrainingWindow:
    def __init__(self):
        self.groups={name:dict(n=0,steps=0,counts=Counter(),ee_displacement_sum=0.,progress_sum=0.)
                     for name in ('overall','0-2cm','2-5cm','5-10cm')}
        self.rewards=Counter()

    def add(self,rows,reward_sums):
        self.rewards.update(reward_sums)
        for row in rows:
            names=['overall']
            if row.get('near_distance_band',-1)>=0:names.append(('0-2cm','2-5cm','5-10cm')[row['near_distance_band']])
            for name in names:
                g=self.groups[name];g['n']+=1;g['steps']+=row['episode_steps']
                g['counts']['success' if row['success'] else row['failure_reason']]+=1
                g['ee_displacement_sum']+=row.get('ee_displacement_m',0.)
                g['progress_sum']+=row.get('initial_to_terminal_progress_m',0.)

    def export(self):
        out={}
        for name,g in self.groups.items():
            n=g['n'];counts=g['counts']
            out[name]=dict(n=n,success_rate=counts['success']/n if n else None,
                counts=dict(counts),failure_rates={k:counts[k]/n if n else None for k in ('timeout','base_contact','tip','nonfoot_contact')},
                physical_failure_rate=sum(counts[k] for k in ('base_contact','tip','nonfoot_contact'))/n if n else None,
                mean_episode_steps=g['steps']/n if n else None,
                mean_ee_displacement_m=g['ee_displacement_sum']/n if n else None,
                mean_initial_to_terminal_progress_m=g['progress_sum']/n if n else None)
        n=self.groups['overall']['n']
        return dict(episodes=out,reward_episode_means={k:v/n if n else None for k,v in self.rewards.items()})
