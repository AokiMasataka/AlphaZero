import numpy as np
from game import Reversi
from self_play import mcts_search, self_play, parallel_self_play


class Model:
    def __call__(self, state):
        return np.random.random(), np.random.random(64)

    def inference_state(self, state):
        return np.random.random(), np.random.random(64)

    def get_value(self, state):
        return np.random.random()


def main():
    model = Model()
    game = Reversi(size=8)

    # play_history = self_play(game=Reversi, model=model, init_dict=dict(size=8), random_play=16)
    # game.play_history_replay(play_history=play_history)

    play_history = parallel_self_play(
        model=model,
        num_searchs=32,
        num_games=4,
        game=Reversi,
        init_dict=dict(size=8),
        random_play=8,
        c_puct=1.0,
        temperature=1.0
    )

    game.play_history_replay(play_history=play_history, n=10)


if __name__ == '__main__':
    main()
