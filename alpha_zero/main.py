import os
import logging
import argparse

from .launch import launch


def load_config(config_path):
    config = dict()
    with open(config_path, 'r') as f:
        config_text = f.read()
    exec(config_text, globals(), config)
    return config, config_text


def set_logger(log_file='../sample.log'):
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)


def main():
    parser = argparse.ArgumentParser(description='Alpha zero')
    parser.add_argument('-c', '--config', default='../configs/config.py', type=str)
    parser.add_argument('-m', '--model', default=None, type=str)
    args = parser.parse_args()
    config, text = load_config(args.config)

    os.makedirs(config['work_dir'], exist_ok=True)
    set_logger(log_file=config['work_dir'] + '/train.log')
    with open(config['work_dir'] + '/config.py', 'w') as f:
        f.write(text)

    self_play_config = config['self_play_config']
    model_config = config['model_config']
    train_config = config['train_config']

    logging.info(f'launch: \n{text}')
    launch(
        self_play_config=self_play_config,
        train_config=train_config,
        model_config=model_config,
        work_dir=config['work_dir'],
        save_play_history=config['save_play_history']
    )


def test():
    from self_play import _self_play
    from model import build_model
    from game import Reversi

    parser = argparse.ArgumentParser(description='Alpha zero')
    parser.add_argument('-c', '--config', default='../configs/config.py', type=str)
    parser.add_argument('-m', '--model', default=None, type=str)
    args = parser.parse_args()

    config, config_text = load_config(config_path=args.config)

    os.makedirs(config["work_dir"], exist_ok=True)
    set_logger(log_file=f'{config["work_dir"]}/train.log')
    logger = logging.getLogger()

    _ = config['model_config'].pop('pretarined_path')
    model = build_model(**config['model_config'])

    play_history = _self_play(
        game_module=Reversi, model=model, init_dict=dict(size=6), num_searchs=16, random_play=16, c_puct=1.0, temperature=1.0
    )

    print(play_history.__len__())


if __name__ == '__main__':
    main()
