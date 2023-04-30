import sys
sys.path.append('D:\\Projects\\alpha_zero')
from alpha_zero.games import Reversi
from alpha_zero.model import ScaleModel
from alpha_zero.engine import MonteCarlo, self_play



def test_monte_carlo(model_config: dict):
    game = Reversi(size=8)
    model = ScaleModel(config=model_config)
    monte_carlo = MonteCarlo(game_module=Reversi, model=model,num_searchs=16)
    monte_carlo(root_obj=game)

    print(f'Success: MonteCarlo')


def test_self_play(model_config: dict):
    model = ScaleModel(config=model_config)
    init_config = dict(size=8)

    play_history = self_play(
        f_model=model,
        b_model=model,
        game_module=Reversi,
        init_dict=init_config,
        num_searchs=16,
        random_play=6,
        c_puct=1.0,
        temperature=1.0
    )

    print(play_history.action_list)
    print(len(play_history))
    print(f'Success: self_play')


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
