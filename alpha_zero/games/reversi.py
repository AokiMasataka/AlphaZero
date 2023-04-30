import copy
import functools
import numpy as np
from .game_base import GAMES, BaseGame, index_to_xy


__all__ = ('Reversi')


@GAMES.register_module
class Reversi(BaseGame):
    def __init__(self, size: int = 8, state=None, player: int = None):
        if state is not None:
            self._state = copy.deepcopy(state)
            self._size = self.state.shape[0]
            self._player = player
        else:
            assert size % 2 == 0
            super(Reversi, self).__init__(state_shape=(size, size), dtype=np.int8)
            self._size = size

            self._state[size // 2, size // 2] = 1
            self._state[size // 2 - 1, size // 2 - 1] = 1
            self._state[size // 2, size // 2 - 1] = -1
            self._state[size // 2 - 1, size // 2] = -1
            self._player = 1

        self._action_space = size * size - 1
        self._pass_action = self._action_space + 1
    
    def __str__(self):
        _str_ = ' '

        for i in range(self._size): _str_ += str(i)
        _str_ += '\n'

        for i in range(self._size):
            _str_ += str(i)
            for j in range(self._size):
                if self.state[i, j] == 1:
                    _str_ += 'O'
                elif self.state[i, j] == -1:
                    _str_ += 'X'
                else:
                    _str_ += '-'
            _str_ += '\n'
        
        return _str_
    
    def get_hash(self) -> bytes:
        return self.state.tobytes()
    
    def encode_state(self) -> np.ndarray:
        player = np.zeros((self._size, self._size), dtype=np.int8)
        return np.stack([(self.state == self._player), (self.state == -self._player), player], axis=0)
    
    def action(self, action: int) -> BaseGame:
        copied_obj = copy.deepcopy(self)
        copied_obj._state = _action_functional(
            state=copied_obj._state.tobytes(),
            action=action,
            shape=copied_obj._state.shape,
            player=copied_obj.player
        )

        copied_obj._player = -copied_obj._player
        return copied_obj
    
    def get_legal_action(self):
        return _get_legal_action_functional(state=self._state.tobytes(), shape=self._state.shape, player=self.player)
    
    def change_player(self) -> BaseGame:
        copied_obj = copy.deepcopy(self)
        copied_obj._state = -copied_obj._state
        copied_obj._player = -copied_obj._player
        return copied_obj
    
    def is_done(self) -> bool:
        if not self.get_legal_action():
            copied_obj = copy.deepcopy(self).change_player()
            if not copied_obj.get_legal_action():
                return True
        return False
    
    def get_winner(self) -> int:
        copied_obj = copy.deepcopy(self).change_player()

        first_hand_legal = self.get_legal_action()
        back_hand_legal = copied_obj.get_legal_action()

        assert first_hand_legal.__len__() == 0
        assert back_hand_legal.__len__() == 0

        result = np.sum(self._state)
        if 0 < result:
            return 1
        elif result < 0:
            return -1
        else:
            return 0


@functools.lru_cache(maxsize=8192)
def _action_functional(state: bytes, action: int, shape: tuple, player: int) -> np.ndarray:
    state = np.frombuffer(buffer=state, dtype=np.int8, count=-1, offset=0).reshape(shape)
    state = state.copy()

    directions = _get_directions(action, size=shape[0])
    for direction in directions:
        stones = [state[i] for i in direction]
        if player in stones and -player in stones:
            idx = stones.index(player)
            stones = stones[:idx]
            if stones and all(i == -player for i in stones):
                for x, y in direction[:idx]:
                    state[x, y] = player

    state[index_to_xy(index=action, n_rows=shape[0], n_cols=shape[0])] = player
    return state


@functools.lru_cache(maxsize=8192)
def _get_legal_action_functional(state: bytes, shape: tuple, player: int):
    state = np.frombuffer(buffer=state, dtype=np.int8, count=-1, offset=0).reshape(shape)
    legal_actions = []
    for action in range(state.size):
        x, y = index_to_xy(index=action, n_rows=shape[0], n_cols=shape[0])
        if state[x, y] != 0:
            continue
        directions = _get_directions(action=action, size=shape[0])
        for direction in directions:
            stones = [state[i] for i in direction]
            if player in stones and -player in stones:
                stones = stones[:stones.index(player)]
                if stones and all(i == -player for i in stones):
                    legal_actions.append(action)
                    break
    return legal_actions


@functools.lru_cache(maxsize=64)
def _get_directions(action, size):
    up = tuple((action // size, y) for y in range(action % size - 1, -1, -1))
    down = tuple((action // size, y) for y in range(action % size + 1, size, 1))
    left = tuple((x, action % size) for x in range(action // size - 1, -1, -1))
    right = tuple((x, action % size) for x in range(action // size + 1, size, 1))

    ul = tuple((x, y) for x, y in zip(range(action // size - 1, -1, -1), range(action % size - 1, -1, -1)))
    ur = tuple((x, y) for x, y in zip(range(action // size + 1, size - 1, 1), range(action % size - 1, -1, -1)))
    dl = tuple((x, y) for x, y in zip(range(action // size - 1, -1, -1), range(action % size + 1, size, 1)))
    dr = tuple((x, y) for x, y in zip(range(action // size + 1, size - 1, 1), range(action % size + 1, size, 1)))
    return (up, down, left, right, ul, ur, dl, dr)
