import sys
sys.path.append('D:\\Projects\\alpha_zero')
from alpha_zero.games import OXGame


def main():
    game = OXGame()

    game = OXGame.action_functional(obj=game, action=0)
    print(game)
    game = OXGame.action_functional(obj=game, action=1)
    print(game)

    

if __name__ == '__main__':
    main()
