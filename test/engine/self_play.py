import sys
sys.path.append('D:\\Projects\\alpha_zero')
import ray
from alpha_zero.games import Reversi
from alpha_zero.model import ScaleModel
from alpha_zero.engine import MonteCarloTreeSearch, self_play



def test_monte_carlo(model_config: dict):
    game = Reversi(size=8)
    model = ScaleModel(config=model_config)
    monte_carlo = MonteCarloTreeSearch(model=model, num_searchs=16)
    monte_carlo(obj=game)

    print(f'Success: MonteCarlo')


def test_self_play(model_config: dict):
    model = ScaleModel(config=model_config)
    init_config = dict(size=8)
    game = Reversi(**init_config)

    work_id = self_play.remote(
        f_model=model,
        b_model=model,
        game=game,
        num_searchs=16,
        random_play=6,
        c_puct=1.0,
        temperature=1.0
    )
    
    play_history = ray.get(work_id)
    
    print(f'history length: {len(play_history)}')
    for i in range(len(play_history)):
        s, a, w = play_history[i]
        print(f'action: {a} - winner: {w}')
        print(dosplay(s))

    print(f'Success: self_play')


def dosplay(s):
    print(s[0].sum(), s[1].sum())
    s = s[0] * 1 + s[1] * -1
    _str_ = ' '
    for i in range(8): _str_ += str(i)
    _str_ += '\n'

    for i in range(8):
        _str_ += str(i)
        for j in range(8):
            if s[i, j] == 1:
                _str_ += 'O'
            elif s[i, j] == -1:
                _str_ += 'X'
            else:
                _str_ += '-'
        _str_ += '\n'
    
    return _str_


def main():
    model_config = dict(
        stem_config=dict(in_channels=2, out_dim=64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),
        block_config=dict(dim=64, eps=1e-6, momentum=0.1, se=True, act_fn='relu'),
        depth=4,
        action_space=65,
        pretarined_path=None
    )

    test_monte_carlo(model_config=model_config)
    test_self_play(model_config=model_config)


if __name__ == '__main__':
    main()
