import os
import sys
sys.path.append(os.getcwd())
from alpha_zero.utils import AvgManager


def test_avg():
    a = AvgManager()

    for i in [0, 1, 2]:
        a += i
    
    assert a() == 1.0
    print('Successfully')


def main():
    test_avg()


if __name__ == '__main__':
    main()
