import sys
sys.path.append('D:\\Projects\\alpha_zero')
from alpha_zero.games import Reversi


def main():
    game = Reversi(size=8)

    for _ in range(100):
        legal_action = game.get_legal_action()
        game = game.action(action=legal_action[0])
        print(game)
        if game.is_done():
            break
    
    print(game)
    print(game.is_done(), game.get_winner())

if __name__ == '__main__':
    main()
