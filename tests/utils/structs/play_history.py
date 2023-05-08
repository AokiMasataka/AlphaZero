import os
import sys
sys.path.append(os.getcwd())

from alpha_zero.utils import PlayHistory


def main():
    p = PlayHistory(
        state=[1 for _ in range(10)],
        actions=[1 for _ in range(10)],
        winner=[1 for _ in range(10)]
    )

    pp = PlayHistory(
        state=[1 for _ in range(5)],
        actions=[1 for _ in range(5)],
        winner=[1 for _ in range(5)]
    )

    p.valid()

    p.append(pp)
    print('Successfully')


if __name__ == '__main__':
    main()

