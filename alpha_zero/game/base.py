import numpy as np
from abc import ABCMeta, abstractmethod


class GameBase(metaclass=ABCMeta):
    def __init__(self, state_shape):
        assert type(state_shape) == tuple
        self.state = np.zeros(state_shape, dtype=np.uint8)
        self.player = 0
        self.done = False

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def is_win(self):
        pass

    @abstractmethod
    def get_legal_action(self):
        pass

    @abstractmethod
    def action(self, action):
        pass

    def get_hash(self):
        return self.state.tobytes()


if __name__ == '__main__':
    pass
