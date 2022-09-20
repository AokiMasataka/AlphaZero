import os
import logging
from copy import deepcopy

from game import GAMES
from data import data_augment
from self_play import parallel_self_play, models_play
from model import trainner, ScaleModel, build_model


def model_evalate(
        old_model, new_model, num_evalate_play, game_module, init_dict, num_searchs, random_play, c_puct, temperature
):

    new_model_winrate = 0
    for i in range(num_evalate_play):
        if i % 2 == 0:
            result = models_play(
                first_model=new_model,
                back_model=old_model,
                game_module=game_module,
                init_dict=init_dict,
                num_searchs=num_searchs,
                random_play=random_play,
                c_puct=c_puct,
                temperature=temperature
            )
        else:
            result = models_play(
                first_model=old_model,
                back_model=new_model,
                game_module=game_module,
                init_dict=init_dict,
                num_searchs=num_searchs,
                random_play=random_play,
                c_puct=c_puct,
                temperature=temperature
            )
        new_model_winrate += abs(result - (i % 2))

    new_model_winrate = new_model_winrate / num_evalate_play
    return new_model_winrate


def launch(self_play_config: dict, train_config: dict, model_config: dict, work_dir, save_play_history):
    self_play_config['game'] = GAMES.get_module(name=self_play_config['game'])
    init_args = self_play_config['init_dict']
    logging.info(f'game name: {self_play_config["game"]} - init args: {init_args}')
    game = self_play_config['game'](**init_args)

    if model_config.get('max_actions', None) is None:
        model_config['max_actions'] = game.max_action()

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

    for gen in range(generation):
        logging.info(f'generation {gen} start selfplay')
        old_model = deepcopy(model)
        if gen == 0:
            play_history = parallel_self_play(model=model, **random_play_config)
        else:
            play_history = parallel_self_play(model=model, **self_play_config)
        logging.info('end selfplay')
        
        logging.info('start data augment')
        play_history = data_augment(
            play_history=play_history,
            hflip=train_config.get('hflip', False),
            vflip=train_config.get('vflip', False),
            max_action=model_config['max_actions'] - 1
        )
        logging.info('end data augment')

        trainner(gen=gen, model=model, play_history=play_history, train_config=train_config)
        new_model_winrate = model_evalate(
            old_model=old_model,
            new_model=model,
            num_evalate_play=32,
            game_module=self_play_config['game'],
            init_dict=init_args,
            num_searchs=self_play_config['num_searchs'],
            random_play=self_play_config['random_play'],
            c_puct=self_play_config['c_puct'],
            temperature=self_play_config['temperature']
        )

        logging.info(msg=f'new model winrate: {new_model_winrate:.4f}')

        save_dir = f'{work_dir}/model_gen{gen}'
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir=save_dir)

        if 0.52 < new_model_winrate:
            pass
        else:
            model = deepcopy(old_model)

        if save_play_history:
            play_history.save_histry(save_path=save_dir + '/play_history.pkl')
