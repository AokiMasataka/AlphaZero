# https://github.com/suragnair/alpha-zero-general/blob/master/MCTS.py

import math
import numpy as np
from ..games.game_base import BaseGame
from ..model.scale_model import ScaleModel


class MonteCarloTreeSearch:
    def __init__(self, model: ScaleModel, num_searchs: int, c_puct: float = 1.0, temperature: float = 1.0):
        self.model = model
        self.num_searchs = num_searchs
        self.cpuct = c_puct
        self.temperature = temperature
        self.init()
    
    def init(self):
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.legal_actions = {}
        
    def __call__(self, obj: BaseGame):
        self.init()
        state = obj.hash

        for _ in range(self.num_searchs):
            self._search(obj=obj)
        
        actions = self.legal_actions[state]
        counts = [self.Nsa[(state, action)] if (state, action) in self.Nsa.keys() else 0 for action in actions]

        counts = [x ** (1. / self.temperature) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def _search(self, obj: BaseGame):
        state = obj.hash

        if obj.is_done():
            return obj.get_winner()

        if state not in self.Ps: # leaf node
            value, policy = self.model.inference_state(obj=obj)
            self.Ps[state] = policy
            self.Ps[state] /= np.sum(self.Ps[state])

            self.legal_actions[state] = obj.get_legal_action()
            self.Ns[state] = 0
            return -value
        
        legal_actions = self.legal_actions[state]
        best_u = -float('inf')
        best_action = -1

        for action in legal_actions:
            u = self._calc_puct(state=state, action=action)

            if best_u < u:
                best_u = u
                best_action = action
        
        action = best_action

        next_obj = obj.action(action=action)
        value = self._search(obj=next_obj)

        if (state, action) in self.Qsa.keys():
            n = self.Nsa[(state, action)] + 1
            self.Qsa[(state, action)] = (self.Nsa[(state, action)] * self.Qsa[(state, action)] + value) / n
            self.Nsa[(state, action)] += 1
        else:
            self.Qsa[(state, action)] = value
            self.Nsa[(state, action)] = 1
        
        self.Ns[state] += 1
        return -value

    def _calc_puct(self, state: str, action: int):
        if (state, action) in self.Qsa.keys():
            _temp = self.Ps[state][action] * math.sqrt(self.Ns[state]) / (1 + self.Nsa[(state, action)])
            return self.Qsa[(state, action)] + self.cpuct * _temp
        else:
            return self.cpuct * self.Ps[state][action] * math.sqrt(self.Ns[state] + 1e-5)  # Q = 0 ?
