from copy import deepcopy
import functools
import math
import numpy as np

from .base import GameBase, index_to_xy, xy_to_index
from .registry import GAMES


@GAMES.register_module
class Reversi(GameBase):
    def __init__(self, size=8, state=None, player=None):
        if state is not None:
            self.state = deepcopy(state)
            self.size = self.state.shape[0]
            self.n_rows = self.state.shape[0]
            self.player = player
        else:
            assert size % 2 == 0
            super(Reversi, self).__init__(state_shape=(size, size), dtype=np.int8)
            self.size = size
            self.n_rows = size

            self.state[size // 2, size // 2] = 1
            self.state[size // 2 - 1, size // 2 - 1] = 1
            self.state[size // 2, size // 2 - 1] = -1
            self.state[size // 2 - 1, size // 2] = -1
            self.player = 1

        self._action_space()

    def __str__(self):
        _str_ = ' '

        for i in range(self.size):
            _str_ += str(i)
        _str_ += '\n'

        for i in range(self.size):
            _str_ += str(i)
            for j in range(self.size):
                if (self.state[i, j] == 1 and self.player == 1) or (self.state[i, j] == -1 and self.player == -1):
                    _str_ += 'O'
                elif (self.state[i, j] == -1 and self.player == 1) or (self.state[i, j] == 1 and self.player == -1):
                    _str_ += 'X'
                else:
                    _str_ += '-'

            _str_ += '\n'
        return _str_

    def _action_space(self):
        self.action_space = self.size * self.size

    def pass_action(self):
        self.state = -self.state
        self.player = -self.player
        return self.action_space

    def get_winner(self, state=None, player=None):
        if state is None:
            result = self.get_winner_functional(state=self.state)
            player = self.player
        else:
            result = self.get_winner_functional(state=state)

        if 0 < result and player == 1 or result < 0 and player == -1:
            return 1
        elif result < 0 and player == 1 or 0 < result and player == -1:
            return -1
        else:
            return 0

    def player_chenge(self):
        self.player = -self.player
        self.state = -self.state

    @staticmethod
    def is_done_functional(state):
        if not Reversi.get_legal_action_functional(state=state):
            if not Reversi.get_legal_action_functional(state=-state):
                return True
            else:
                return False
        else:
            return False

    @staticmethod
    def get_winner_functional(state=None):
        first_hand_legal = Reversi.get_legal_action_functional(state=state)
        back_hand_legal = Reversi.get_legal_action_functional(state=-state)

        assert first_hand_legal.__len__() == 0
        assert back_hand_legal.__len__() == 0

        result = np.sum(state)
        if 0 < result:
            return 1
        elif result < 0:
            return -1
        else:
            return 0

    @staticmethod
    def get_legal_action_functional(state):
        byte_state = state.tobytes()
        return Reversi._get_legal_action_functional(state=byte_state, shape=state.shape)

    @staticmethod
    @functools.lru_cache(maxsize=8192)
    def _get_legal_action_functional(state, shape):
        state = np.frombuffer(buffer=state, dtype=np.int8, count=-1, offset=0).reshape(shape)
        legal_actions = []
        for action in range(state.size):
            x, y = index_to_xy(index=action, n_rows=state.shape[0], n_cols=state.shape[0])
            if state[x, y] != 0:
                continue
            directions = get_directions(action=action, size=state.shape[0])
            for direction in directions:
                stones = [state[i] for i in direction]
                if 1 in stones and -1 in stones:
                    stones = stones[:stones.index(1)]
                    if stones and all(i == -1 for i in stones):
                        legal_actions.append(action)
                        break
        return legal_actions

    @staticmethod
    def get_next_state(state, action):
        next_state = deepcopy(state)

        directions = get_directions(action, size=state.shape[0])
        for direction in directions:
            stones = [state[i] for i in direction]
            if 1 in stones and -1 in stones:
                idx = stones.index(1)
                stones = stones[:idx]
                if stones and all(i == -1 for i in stones):
                    for x, y in direction[:idx]:
                        next_state[x, y] = 1

        next_state[index_to_xy(index=action, n_rows=state.shape[0], n_cols=state.shape[0])] = 1
        return -next_state

    @staticmethod
    def encode_state(state):
        return np.stack([(state == 1), (state == -1)], axis=0)


@functools.lru_cache(maxsize=1024)
def get_directions(action, size):
    up = tuple((action // size, y) for y in range(action % size - 1, -1, -1))
    down = tuple((action // size, y) for y in range(action % size + 1, size, 1))
    left = tuple((x, action % size) for x in range(action // size - 1, -1, -1))
    right = tuple((x, action % size) for x in range(action // size + 1, size, 1))

    ul = tuple((x, y) for x, y in zip(range(action // size - 1, -1, -1), range(action % size - 1, -1, -1)))
    ur = tuple((x, y) for x, y in zip(range(action // size + 1, size - 1, 1), range(action % size - 1, -1, -1)))
    dl = tuple((x, y) for x, y in zip(range(action // size - 1, -1, -1), range(action % size + 1, size, 1)))
    dr = tuple((x, y) for x, y in zip(range(action // size + 1, size - 1, 1), range(action % size + 1, size, 1)))
    return tuple([up, down, left, right, ul, ur, dl, dr])
