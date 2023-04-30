import copy
import numpy as np
from .game_base import GameBase, index_to_xy


class OXGame(GameBase):
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
    
    def get_hash(self):
        return self._state.tobytes()
    
    def encode_state(self):
        return np.stack([(self.state == self.player), (self.state == -self.player)], axis=0)
    