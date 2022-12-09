import os
import argparse
from utils import load_config, set_logger
from engine import train


def main():
    parser = argparse.ArgumentParser(description='Alpha zero')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-c', '--config', default='../configs/config.py', type=str)
    parser.add_argument('-m', '--model', default=None, type=str)
    args = parser.parse_args()

    config, text = load_config(args.config)
    with open(config['work_dir'] + '/config.py', 'w') as f:
        f.write(text)
    
    if args.train:
        set_logger(log_file=os.path.join(config['work_dir'], 'train.log'))
        train(config=args.config)
    else:
        pass