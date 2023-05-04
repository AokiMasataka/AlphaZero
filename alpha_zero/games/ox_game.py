import copy
import itertools
import numpy as np
from .game_base import BaseGame, index_to_xy


class OXGame(BaseGame):
    def __init__(self, state=None, player: int = None):
        if state is not None:
            assert player is not None
            self._state = copy.deepcopy(state)
            self._size = 3
            self._player = player
        else:
            super(OXGame, self).__init__(state_shape=(3, 3), dtype=np.int8)
            self._size = 3
            self._player = 1

        self._action_space = 8
        self._pass_action = self._action_space + 1
    
    def __str__(self):
        _str_ = ' '
        for i in range(self._size): _str_ += str(i)
        _str_ += '\n'

        for i in range(self._size):
            _str_ += str(i)
            for j in range(self._size):
                if self._state[i, j] == self._player:
                    _str_ += 'O'
                elif self._state[i, j] == -self._player:
                    _str_ += 'X'
                else:
                    _str_ += '-'
            _str_ += '\n'
        return _str_
    
    def encode_state(self):
        return np.stack([(self.state == 1), (self.state == -1)], axis=0)
    
    def action(self, action: int) -> BaseGame:
        x, y = index_to_xy(index=action, n_rows=self._size, n_cols=self._size)
        if self._state[x, y] == 0:
            copied_obj = copy.deepcopy(self)
            copied_obj._state[x, y] = 1
            copied_obj = copied_obj.change_player()
        
            return copied_obj
        return None
    
    def get_legal_action(self):
        legal_actions = []
        for x, y in itertools.product(range(self._size), range(self._size)):
            if self._state[x, y] == 0:
                legal_actions[x * self._size + y]
        return legal_actions
    
    def change_player(self) -> BaseGame:
        copied_obj = copy.deepcopy(self)
        copied_obj._state = -copied_obj._state
        copied_obj._player = -copied_obj._player
        return copied_obj

    def is_done(self) -> bool:
        if self.get_winner() != 0:
            return True
        
        if not self.get_legal_action():
            copied_obj = copy.deepcopy(self).change_player()
            if not copied_obj.get_legal_action():
                return True
        return False
    
    def get_winner(self):
        for i in range(self._size):
            _temp = np.sum(self._state[i, :])
            if  _temp == self._size or _temp == -self._size:
                return _temp

            _temp = np.sum(self._state[:, i])
            if  _temp == self._size or _temp == -self._size:
                return _temp
        
        _temp = np.sum(np.diag(self._state))
        if _temp == self._size or _temp == -self._size:
            return _temp
        
        _temp = np.sum(np.diag(self._state[:, ::-1]))
        if _temp == self._size or _temp == -self._size:
            return _temp
        
        return 0