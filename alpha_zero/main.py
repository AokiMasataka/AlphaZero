from game import Reversi


def main():
    game = Reversi(size=8)
    print(game)
    state = game.get_next_state(action=20)
    print(state)


if __name__ == '__main__':
    main()
