from copy import deepcopy
import numpy as np
from scipy.ndimage import shift

from .base import GameBase
from .registry import GAMES


@GAMES.register_module
class Reversi(GameBase):
    def __init__(self, size=8, state=None):
        if state is None:
            assert size % 2 == 0
            super(Reversi, self).__init__(state_shape=(2, size, size))
            self.size = size
            half = size // 2
            self.state[0, half, half] = 1
            self.state[0, half - 1, half - 1] = 1

            self.state[1, half, half - 1] = 1
            self.state[1, half - 1, half] = 1
        else:
            self.state = state
            self.size = self.state.shape[1]

    def __call__(self):
        return deepcopy(self.state)

    def __str__(self):
        _str_ = ' '

        for i in range(self.size):
            _str_ += str(i)
        _str_ += '\n'

        for i in range(self.size):
            _str_ += str(i)
            for j in range(self.size):
                if self.state[self.player, i, j] == 1:
                    _str_ += 'O'
                elif self.state[1 - self.player, i, j] == 1:
                    _str_ += 'X'
                else:
                    _str_ += '-'
            _str_ += '\n'
        return _str_

    def xy_to_action(self, x, y):
        return x * self.size + y

    def is_done(self):
        if self.done:
            return self.done

        if np.sum(self.state) == 64:
            self.done = True
            return True

        state = deepcopy(self.state)
        if not self.get_legal_action(state=state):
            if not self.get_legal_action(state=state[::-1]):
                return True
        return False

    def is_win(self):
        if np.sum(self.state[0]) < np.sum(self.state[1]):
            return False
        else:
            return True

    def winner(self):
        if self.state[1 - self.player].sum() < self.state[self.player].sum():
            return 1
        elif self.state[self.player].sum() < self.state[1 - self.player].sum():
            return -1
        else:
            return 0

    def play_chenge(self):
        self.state = self.state[::-1]
        self.player = 1 - self.player

    def action(self, action):
        x = action // self.size
        y = action % self.size
        self.state[0, x, y] = 1

        for dx, dy in ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)):
            for i in range(1, self.size):
                pos_x, pos_y = i * dx + x, i * dy + y
                if 0 <= pos_x < self.size and 0 <= pos_y < self.size:
                    if self.state[1, pos_x, pos_y]:
                        pass
                    elif self.state[0, pos_x, pos_y]:
                        if 1 < i:
                            for n in range(i):
                                self.state[0, n * dx + x, n * dy + y] = 1
                                self.state[1, n * dx + x, n * dy + y] = 0
                            break
                        else:
                            break
                    else:
                        break
                else:
                    break

        self.play_chenge()

    def get_legal_action(self, state=None):
        # loop 1000: 0.714038610458374 s
        if state is None:
            player_state = self.state[0]
            opponent_state = self.state[1]
        else:
            player_state = state[0]
            opponent_state = state[1]

        player_state_satck = np.hstack((
            player_state,
            np.rot90(m=player_state, k=1, axes=(1, 0)),
            np.rot90(m=player_state, k=2, axes=(1, 0)),
            np.rot90(m=player_state, k=3, axes=(1, 0))
        ))

        opponent_state_stack = np.hstack((
            opponent_state,
            np.rot90(m=opponent_state, k=1, axes=(1, 0)),
            np.rot90(m=opponent_state, k=2, axes=(1, 0)),
            np.rot90(m=opponent_state, k=3, axes=(1, 0))
        ))

        player_state_satck_1 = deepcopy(player_state_satck)
        opponent_state_stack_1 = deepcopy(opponent_state_stack)

        blank = ~(player_state_satck | opponent_state_stack)

        temp = opponent_state_stack & shift(input=player_state_satck, shift=(0, 1), cval=0)
        for _ in range(self.size - 3):
            temp |= opponent_state_stack & shift(input=temp, shift=(0, 1), cval=0)
        legal_state = blank & shift(input=temp, shift=(0, 1), cval=0)

        temp = opponent_state_stack_1 & shift(input=player_state_satck_1, shift=(1, 1), cval=0)
        for _ in range(self.size - 3):
            temp |= opponent_state_stack_1 & shift(input=temp, shift=(1, 1), cval=0)
        legal_state_diagonal = blank & shift(input=temp, shift=(1, 1), cval=0)

        legal_state_0, legal_state_1, legal_state_2, legal_state_3 = np.hsplit(legal_state, 4)
        legal_state_1 = np.rot90(m=legal_state_1, k=1, axes=(0, 1))
        legal_state_2 = np.rot90(m=legal_state_2, k=2, axes=(0, 1))
        legal_state_3 = np.rot90(m=legal_state_3, k=3, axes=(0, 1))
        legal_state = legal_state_0 | legal_state_1 | legal_state_2 | legal_state_3

        legal_state_0, legal_state_1, legal_state_2, legal_state_3 = np.hsplit(legal_state_diagonal, 4)
        legal_state_1 = np.rot90(m=legal_state_1, k=1, axes=(0, 1))
        legal_state_2 = np.rot90(m=legal_state_2, k=2, axes=(0, 1))
        legal_state_3 = np.rot90(m=legal_state_3, k=3, axes=(0, 1))
        legal_state |= legal_state_0 | legal_state_1 | legal_state_2 | legal_state_3

        legal_actions = []
        for index, legal in enumerate(legal_state.reshape(-1)):
            if legal:
                legal_actions.append(index)

        return legal_actions

    def random_action(self):
        if self.is_done():
            return -1
        else:
            legal_actions = self.get_legal_action()
            if legal_actions:
                action = np.random.choice(a=legal_actions, size=1)
                self.action(action=action)
                return action
            else:
                self.play_chenge()
                return self.pass_action()

    def pass_action(self):
        return self.state.shape[1] * self.state.shape[1]

    def max_action(self):
        return self.state.shape[1] * self.state.shape[1] + 1

    @staticmethod
    def get_next_state(state, action):
        size = state.shape[1]
        x = action // size
        y = action % size
        state[0, x, y] = 1

        for dx, dy in ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)):
            for i in range(1, size):
                pos_x, pos_y = i * dx + x, i * dy + y
                if 0 <= pos_x < size and 0 <= pos_y < size:
                    if state[1, pos_x, pos_y]:
                        pass
                    elif state[0, pos_x, pos_y]:
                        if 1 < i:
                            for n in range(i):
                                state[0, n * dx + x, n * dy + y] = 1
                                state[1, n * dx + x, n * dy + y] = 0
                            break
                        else:
                            break
                    else:
                        break
                else:
                    break

        return state[::-1]

    def play_history_replay(self, play_history, n=None):
        if n is None:
            n = play_history.__len__()
        for step, _ in zip(range(play_history.__len__()), range(n)):
            state, action, _ = play_history[step]
            x = action // self.size
            y = action % self.size

            _str_ = ''
            for i in range(self.size):
                for j in range(self.size):
                    if i == x and j == y:
                        _str_ += 'a'
                    else:
                        if state[0, i, j] == 1:
                            _str_ += 'O'
                        elif state[1, i, j] == 1:
                            _str_ += 'X'
                        else:
                            _str_ += '-'
                _str_ += '\n'
            print(_str_)

    def api(self, x=None, y=None):
        legal_actions = self.get_legal_action()
        _str_ = ''
        for legal_action in legal_actions:
            _x = legal_action // self.size
            _y = legal_action % self.size
            _str_ += f'({_x}, {_y}) '
        print(_str_)
        if x is None:
            x = int(input('x ='))
            y = int(input('y ='))

        action = self.xy_to_action(x=x, y=y)
        self.action(action=action)
        return action
