import numpy as np
from abc import ABCMeta, abstractmethod


class Registry:
    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def register_module(self, module):
        self._module_dict[module.__name__] = module
        return module

    def get_module(self, name):
        return self._module_dict[name]


GAMES = Registry(name='games')


class GameBase(metaclass=ABCMeta):
    def __init__(self, state_shape, dtype):
        self.state = np.zeros(state_shape, dtype=dtype)
        self.player = 0
        self.action_space = -1

    def get_hash(self):
        return self.state.tobytes()

    def is_done(self, state=None):
        if state is None:
            return self.is_done_functional(state=self.state)
        else:
            return self.is_done_functional(state=state)

    def get_winner(self, state=None):
        if state is None:
            return self.get_winner_functional(state=self.state)
        else:
            return self.get_winner_functional(state=state)

    def get_legal_action(self, state=None):
        if state is None:
            return self.get_legal_action_functional(state=self.state)
        else:
            return self.get_legal_action_functional(state=state)

    def action(self, action, state=None):
        if state is None:
            self.state = self.get_next_state(state=self.state, action=action)
        else:
            self.state = self.get_next_state(state=state, action=action)

        self.player = -self.player

    @abstractmethod
    def _action_space(self):
        pass

    @abstractmethod
    def pass_action(self):
        pass

    @staticmethod
    @abstractmethod
    def is_done_functional(state):
        pass

    @staticmethod
    @abstractmethod
    def get_winner_functional(state):
        pass

    @staticmethod
    @abstractmethod
    def get_legal_action_functional(state):
        pass

    @staticmethod
    @abstractmethod
    def get_next_state(state, action):
        pass

    @staticmethod
    @abstractmethod
    def encode_state(state):
        pass


def xy_to_index(row, col, n_rows):
    return n_rows * row + col


def index_to_xy(index, n_rows, n_cols):
    return index // n_rows, index % n_cols
