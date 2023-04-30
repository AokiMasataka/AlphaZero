import numpy as np


__all__ = ['GAMES', 'GameBase', 'xy_to_index', 'index_to_xy']


class GameBase:
    def __init__(self, state_shape: tuple, dtype=np.int8):
        self._state = np.zeros(state_shape, dtype=dtype)
        self._player = None
    
    @property
    def action_space(self):
        return self._action_space
    
    @property
    def pass_action(self):
        return self._pass_action
    
    @property
    def state(self):
        return self._state
    
    @property
    def player(self):
        return self._player
    
    def set_state(self, state):
        self._state = state
    
    def get_hash(self):
        raise NotImplementedError('must be implemented')
    
    def encode_state(self):
        raise NotImplementedError('must be implemented')
    
    def action(self, action: int):
        raise NotImplementedError('must be implemented')
    
    def get_legal_action(self):
        raise NotImplementedError('must be implemented')
    
    def change_player(self):
        raise NotImplementedError('must be implemented')

    def is_done(self):
        raise NotImplementedError('must be implemented')
    
    def get_winner(self):
        raise NotImplementedError('must be implemented')


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module):
        self._module_dict[module.__name__] = module
        return module

    def get_module(self, name: str) -> GameBase:
        return self._module_dict[name]


GAMES = Registry(name='games')


def xy_to_index(row, col, n_rows):
    return n_rows * row + col


def index_to_xy(index, n_rows, n_cols):
    return index // n_rows, index % n_cols