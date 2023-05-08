import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from alpha_zero.games import Reversi


def test_action():
    game = Reversi(size=8)

    for _ in range(4):
        legal_actions = game.get_legal_action()
        game.action(legal_actions[0], inplace=True)
        assert not game.is_done()

    print('Successfully: action')


def test_done_winner():
    game = Reversi(size=8)

    game._state = np.abs(game._state)
    assert game.is_done()
    assert game.get_winner() == 1
    game._state = -game._state
    assert game.is_done()
    assert game.get_winner() == -1
    print('Successfully: done winner')


def main():
    test_action()
    test_done_winner()


if __name__ == '__main__':
    main()
