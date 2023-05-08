import copy
import functools
import numpy as np
from .game import GAMES, Game, index_to_xy


@GAMES.register_module
class Reversi(Game):
    def __init__(self, size: int):
        super().__init__(init_shape=(size, size), dtype=np.int8)
        self._state[size // 2, size // 2] = 1
        self._state[size // 2 - 1, size // 2 - 1] = 1
        self._state[size // 2, size // 2 - 1] = -1
        self._state[size // 2 - 1, size // 2] = -1
        self._player = 1

        self._size = size
        self._action_space = size * size - 1
        self._pass_action = self._action_space + 1

    def __repr__(self) -> str:
        _str = ' ' + ''.join([str(i) for i in range(self._size)]) + '\n'

        for i in range(self._size):
            _str += str(i)
            for j in range(self._size):
                if self._state[i, j] == self._player: _str += 'O'
                elif self._state[i, j] == -self._player: _str += 'X'
                else: _str += '-'
            _str += '\n'
        return _str
    
    def encode_state(self):
        return np.stack([self._state == 1, self._state == -1], dtype=np.float32)
    
    def get_legal_action(self) -> list[int]:
        return _get_legal_action_functional(bytes_state=self._state.tobytes(), shape=self._state.shape)

    def action(self, action: int, inplace: bool = False) -> Game:
        if inplace:
            self._action_inplace(action=action)
        else:
            copied_game = copy.deepcopy(self)
            copied_game._action_inplace(action=action)
            return copied_game
    
    def _action_inplace(self, action: int) -> None:
        if action != self.pass_action:
            self._state = _action_functional(
                bytes_state=self._state.tobytes(),
                action=action,
                shape=self._state.shape
            )
        self.change_player(inplace=True)
    
    def change_player(self, inplace: bool = False) -> Game:
        if inplace:
            self._state = -self._state
            self._player = -self._player
        else:
            copied_game = copy.deepcopy(self)
            copied_game._state = -copied_game._state
            copied_game._player = -copied_game._player
            return copied_game
    
    def is_done(self) -> bool:
        if not self.get_legal_action():
            copied_game = copy.deepcopy(self).change_player()
            if not copied_game.get_legal_action():
                return True
        return False
    
    def get_winner(self):
        copied_game = copy.deepcopy(self).change_player()

        first_hand_legal = self.get_legal_action()
        back_hand_legal = copied_game.get_legal_action()

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
def _get_legal_action_functional(bytes_state: bytes, shape: tuple):
    state = np.frombuffer(buffer=bytes_state, dtype=np.int8, count=-1, offset=0).reshape(shape)
    legal_actions = []
    for action in range(state.size):
        x, y = index_to_xy(index=action, n_rows=shape[0], n_cols=shape[0])
        if state[x, y] != 0:
            continue
        directions = _get_directions(action=action, size=shape[0])
        for direction in directions:
            stones = [state[i] for i in direction]
            if 1 in stones and -1 in stones:
                stones = stones[:stones.index(1)]
                if stones and all(i == -1 for i in stones):
                    legal_actions.append(action)
                    break
    return legal_actions


@functools.lru_cache(maxsize=8192)
def _action_functional(bytes_state: bytes, action: int, shape: tuple) -> np.ndarray:
    state = np.frombuffer(buffer=bytes_state, dtype=np.int8, count=-1, offset=0).reshape(shape)
    state = state.copy()

    directions = _get_directions(action, size=shape[0])
    for direction in directions:
        stones = [state[i] for i in direction]
        if 1 in stones and -1 in stones:
            idx = stones.index(1)
            stones = stones[:idx]
            if stones and all(i == -1 for i in stones):
                for x, y in direction[:idx]:
                    state[x, y] = 1

    state[index_to_xy(index=action, n_rows=state.shape[0], n_cols=state.shape[0])] = 1
    return state


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
