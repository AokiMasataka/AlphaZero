import argparse
from model import ScaleModel
from game import GAMES
from self_play import mcts_search


def load_config(config_path):
    config = dict()
    with open(config_path, 'r') as f:
        config_text = f.read()
    exec(config_text, globals(), config)
    return config, config_text


class PlayState:
    def __init__(self, model, game_module, init_dict):
        self.model = model
        self.game_module = game_module
        self.init_dict = init_dict

        self.game = game_module(**init_dict)

    def ai_action(self, num_searchs=128, c_puct=0.5, temperature=0.1):
        legal_action = self.game.get_legal_action()
        if not legal_action:
            self.game.play_chenge()
            action = None
            print('AI: Pass')
        elif legal_action.__len__() == 1:
            action = legal_action[0]
        else:
            action_index, value = mcts_search(
                root_state=self.game.state,
                model=self.model,
                game_module=self.game_module,
                init_dict=self.init_dict,
                num_searchs=num_searchs,
                c_puct=c_puct,
                temperature=temperature,
                return_value=True
            )
            action = legal_action[action_index]
            print(f'AI value: {value:.4f}')

        if action is not None:
            self.game.action(action=action)

    def player_action(self, legal_actions, action=None, x=None, y=None):
        if not legal_actions:
            self.game.play_chenge()
            print('Player: Pass')
        else:
            if action is not None:
                self.game.action(action=action)
            else:
                self.game.api(legal_actions)


def play_cui(model, game_module, init_dict, num_searchs, c_puct):
    play_state = PlayState(model=model, game_module=game_module, init_dict=init_dict)

    while not play_state.game.is_done():
        print(play_state.game)
        if not play_state.game.player:
            play_state.game.api()
        else:
            play_state.ai_action(num_searchs=num_searchs, c_puct=c_puct)

    if play_state.game.winner() == 1:
        print(f'winner: O')
    elif play_state.game.winner() == -1:
        print(f'winner: X')
    else:
        print('Draw')


def main():
    parser = argparse.ArgumentParser(description='Alpha zero')
    parser.add_argument('-c', '--config', default=None, type=str, help='path to config')
    parser.add_argument('-m', '--model_path', default=None, type=str, help='path to config')
    parser.add_argument('-g', '--gui', action='store_true', help='is use gui')
    parser.add_argument('-s', '--searchs', default=-1, type=int, help='mcts num_searchs')
    args = parser.parse_args()

    assert args.model_path is not None, f'--model_path expect arg'
    assert args.config is not None

    config, _ = load_config(args.config)
    game_config = config['self_play_config']

    if args.searchs != -1:
        game_config['num_searchs'] = args.searchs

    model = ScaleModel.from_pretrained(load_dir=args.model_path).eval()
    game_module = GAMES.get_module(name=game_config['game'])
    init_dict = game_config['init_dict']
    num_searchs = game_config['num_searchs']
    c_puct = game_config['c_puct']

    if args.gui:
        pass
    else:
        play_cui(model=model, game_module=game_module, init_dict=init_dict, num_searchs=num_searchs, c_puct=c_puct)


if __name__ == '__main__':
    main()
