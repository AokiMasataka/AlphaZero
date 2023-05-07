import numpy as np


__all__ = ['GAMES', 'Game', 'xy_to_index', 'index_to_xy']


class Game:
    def __init__(self, init_shape: tuple, dtype=np.int8):
        self._state = np.zeros(init_shape, dtype=dtype)
        self._player = None
        self._action_space = None
        self._pass_action = None 

    @property
    def action_space(self) -> int:
        return self._action_space

    @property
    def pass_action(self) -> int:
        return self._pass_action

    @property
    def state(self) -> np.ndarray:
        return self._state

    @property
    def player(self) -> int:
        return self._player
    
    @property
    def hash(self) -> bytes:
        return self._state.tobytes()

    def set_state(self, state: np.ndarray):
        self._state = state

    def encode_state(self) -> np.ndarray:
        raise NotImplementedError('must be implemented')

    def action(self, action: int):
        raise NotImplementedError('must be implemented')

    def get_legal_action(self) -> list[int]:
        raise NotImplementedError('must be implemented')

    def change_player(self):
        raise NotImplementedError('must be implemented')

    def is_done(self) -> bool:
        raise NotImplementedError('must be implemented')

    def get_winner(self) -> int:
        raise NotImplementedError('must be implemented')


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module: Game) -> Game:
        self._module_dict[module.__name__] = module
        return module

    def get_module(self, name: str) -> Game:
        return self._module_dict[name]


GAMES = Registry(name='games')


def xy_to_index(row, col, n_rows):
    return n_rows * row + col


def index_to_xy(index, n_rows, n_cols):
    return index // n_rows, index % n_cols