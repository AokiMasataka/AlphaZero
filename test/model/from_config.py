import sys
sys.path.append('D:\\Projects\\alpha_zero')
from alpha_zero.model import ScaleModel



def main():
    config_path = './configs/debug_config.py'
    model = ScaleModel.from_config(config_path)


if __name__ == '__main__':
    main()