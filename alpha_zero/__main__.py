import os
import argparse
from .utility import load_config, set_logger
from .engine import train


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='../configs/config.py', type=str)
    parser.add_argument('-m', '--model', default=None, type=str)
    parser.add_argument('-p', '--play', action='store_true')
    args = parser.parse_args()

    config, text = load_config(args.config)

    if args.play == False:
        os.makedirs(config['work_dir'], exist_ok=True)
        with open(config['work_dir'] + '/config.py', 'w') as f:
            f.write(text)
        set_logger(log_file=os.path.join(config['work_dir'], 'train.log'))
        train(config=config, save_play_history=True)
    else:
        pass


if __name__ == '__main__':
    main()
