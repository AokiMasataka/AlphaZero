import sys
sys.path.append('D:\\Projects\\alpha_zero')
from alpha_zero.games import OXGame


def main():
    game = OXGame()

    for _ in range(32):
        legal_actions = game.get_legal_action()
        game = game.action(legal_actions[0])
        print(game)
        if game.is_done():
            break

    print(game)
    print(game.get_winner())

    

if __name__ == '__main__':
    main()
