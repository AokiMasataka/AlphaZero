import os
import logging
from copy import deepcopy
from torch import cuda

from game import GAMES
from data import data_augment
from self_play import parallel_self_play
from model import trainner, ScaleModel, build_model


def model_evalate(new_model, old_model, evalate_config):
    num_evalate_play = evalate_config['num_games'] * 2

    _, f_winner = parallel_self_play(f_model=new_model, b_model=old_model, **evalate_config)
    _, b_winner = parallel_self_play(f_model=old_model, b_model=new_model, **evalate_config)
    b_winner = list(map(lambda x: -x, b_winner))

    winarte = sum(f_winner + b_winner)
    f_winrate = 0.5 + (sum(f_winner) / (num_evalate_play // 2))
    b_winrate = 0.5 + (sum(b_winner) / (num_evalate_play // 2))
    return 0.5 + (winarte / num_evalate_play), f_winrate, b_winrate


def launch(self_play_config: dict, train_config: dict, model_config: dict, work_dir, save_play_history):
    self_play_config['game'] = GAMES.get_module(name=self_play_config['game'])
    init_args = self_play_config['init_dict']
    logging.info(f'game name: {self_play_config["game"]} - init args: {init_args}')
    game = self_play_config['game'](**init_args)

    if model_config.get('action_space', None) is None:
        model_config['action_space'] = game.action_space

    if 'pretarined_path' not in model_config.keys():
        model_config['pretarined_path'] = None

    pretarined_path = model_config.pop('pretarined_path')
    if pretarined_path is not None:
        model = ScaleModel.from_pretrained(load_dir=pretarined_path)
    else:
        model = build_model(**model_config)

    model_config_str = str(model.config)
    model_config_str = model_config_str.replace('\n', '\n\t')
    logging.info(f'build model: \n\t{model_config_str}')

    generation = self_play_config.pop('generation')

    random_play_config = deepcopy(self_play_config)
    random_play_config['random_play'] = 1000

    evalate_config = deepcopy(self_play_config)
    evalate_config['num_games'] = 50 // 2

    if (not cuda.is_available()) and train_config.get('device', 'cpu') == 'cuda':
        logging.info(msg='device param in train_config is "cuda", but cuda is not available. use cpu')
        train_config['device'] = 'cpu'

    for gen in range(1, generation):
        logging.info(f'generation {gen} start selfplay')
        old_model = deepcopy(model)
        if gen == 1:
            play_history, _ = parallel_self_play(f_model=model, b_model=model, **random_play_config)
        else:
            play_history, _ = parallel_self_play(f_model=model, b_model=model, **self_play_config)
        logging.info('end selfplay')

        play_history = data_augment(
            play_history=play_history,
            hflip=train_config.get('hflip', False),
            vflip=train_config.get('vflip', False),
            max_action=model_config['action_space'] - 1
        )

        trainner(gen=gen - 1, model=model, play_history=play_history, train_config=train_config)
        winrate, f_winrate, b_winrate = model_evalate(
            new_model=model, old_model=old_model, evalate_config=evalate_config)

        msg = f'new model winrate: {winrate:.4f}'
        msg += f' - fisrt hand winrate: {f_winrate:.4f}'
        msg += f' - back hand winrate: {b_winrate: 4f}\n'
        logging.info(msg=msg)

        save_dir = f'{work_dir}/model_gen{gen}'
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir=save_dir)

        if winrate < 0.52:
            model = deepcopy(old_model)

        if save_play_history:
            play_history.save_histry(save_path=save_dir + '/play_history.pkl')
