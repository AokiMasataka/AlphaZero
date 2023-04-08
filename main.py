import os
import argparse
from alpha_zero.utils import load_config, set_logger
from alpha_zero.engine import train


def main():
    parser = argparse.ArgumentParser(description='Alpha zero')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-c', '--config', default='../configs/config.py', type=str)
    parser.add_argument('-m', '--model', default=None, type=str)
    args = parser.parse_args()

    config, text = load_config(args.config)
    
    if args.train:
        os.makedirs(config['work_dir'], exist_ok=True)
        with open(config['work_dir'] + '/config.py', 'w') as f:
            f.write(text)
        set_logger(log_file=os.path.join(config['work_dir'], 'train.log'))
        train(config=config, save_play_history=True)
    else:
        pass


if __name__ == '__main__':
    main()
