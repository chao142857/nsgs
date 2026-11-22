"""Training-only curriculum; decisions at the end of an update apply next update."""
from collections import deque
import numpy as np

DEFAULTS = dict(success_threshold=.6, window_updates=50, minimum_episodes=200, newest_mass=.5)


class ManualCurriculum:
    def __init__(self, regions, **settings):
        self.settings = dict(DEFAULTS, **settings)
        s = self.settings
        if not 0 <= s['success_threshold'] <= 1 or not 0 < s['newest_mass'] < 1:
            raise ValueError('Invalid curriculum probabilities')
        if s['window_updates'] < 1 or s['minimum_episodes'] < 1:
            raise ValueError('Invalid curriculum window')
        self.regions = np.asarray(regions)
        if set(self.regions.tolist()) != set(range(4)):
            raise ValueError('All four regions required')
        self.stage = 0
        self.history = deque()
        self.events = []
        self.last_update = 0

    def probabilities(self):
        p = np.zeros(len(self.regions), dtype=np.float64)
        newest = self.regions == self.stage
        p[newest] = (1. if self.stage == 0 else self.settings['newest_mass']) / newest.sum()
        earlier = self.regions < self.stage
        if earlier.any():
            p[earlier] = (1-self.settings['newest_mass']) / earlier.sum()
        return p

    def record(self, update, region, success):
        self.history.append((int(update), int(region), bool(success)))

    def end_update(self, update):
        if update <= self.last_update:
            raise ValueError('Updates must increase')
        self.last_update = update
        first = update-self.settings['window_updates']+1
        while self.history and self.history[0][0] < first:
            self.history.popleft()
        eligible = [v for u,r,v in self.history if first <= u <= update and r == self.stage]
        if self.stage == 3 or len(eligible) < self.settings['minimum_episodes']:
            return None
        rate = sum(eligible)/len(eligible)
        if rate < self.settings['success_threshold']:
            return None
        self.stage += 1
        event = dict(update=update+1, trigger_update=update, opened_region=self.stage,
                     success_rate=rate, episodes=len(eligible))
        self.events.append(event)
        return event

    def state_dict(self):
        return dict(settings=self.settings, stage=self.stage, history=list(self.history),
                    events=self.events, last_update=self.last_update)
