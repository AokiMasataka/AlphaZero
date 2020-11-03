import numpy as np
import random
import config as conf


class Game:
    def __init__(self, state=None):
        if state == None:
            self.state = np.zeros((3, conf.BOARD_SIZE, conf.BOARD_SIZE), dtype=np.int)
            half = int(conf.BOARD_SIZE / 2)
            self.state[0, half - 1, half - 1] = self.state[0, half, half] = 1
            self.state[1, half - 1, half] = self.state[1, half, half - 1] = 1
        else:
            self.state = state

    def isDone(self):
        if not len(self.getLegalAction()):
            self.changePlayer()
            if not len(self.getLegalAction()):
                self.changePlayer()
                return True
            self.changePlayer()
        return False

    def getWinner(self):
        if np.sum(self.state[0, :, :]) < np.sum(self.state[1, :, :]):
            return -1
        elif np.sum(self.state[0, :, :]) > np.sum(self.state[1, :, :]):
            return 1
        else:
            return 0

    def getStone(self):
        return np.sum(self.state[0, :, :]) + np.sum(self.state[1, :, :])

    def getPlayer(self):
        return int(self.state[2][0][0])

    def getLegalAction(self):
        actions = []
        for i in range(conf.BOARD_SIZE):
            for j in range(conf.BOARD_SIZE):
                if self.state[0][i][j] == 0 and self.state[1][i][j] == 0:
                    if self.legalAction(i, j):
                        actions.append(i + j * conf.BOARD_SIZE)
        return actions

    def action(self, action):
        x = int(action % conf.BOARD_SIZE)
        y = int(action / conf.BOARD_SIZE)
        self.flip(x, y)
        self.changePlayer()

    def changePlayer(self):
        self.state[2, :, :] = 1 - self.state[2][0][0]

    def flip(self, x, y):
        player = self.getPlayer()

        for dx, dy in ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)):
            for i in range(1, conf.BOARD_SIZE):
                posX, posY = i * dx + x, i * dy + y
                if 0 <= posX < conf.BOARD_SIZE and 0 <= posY < conf.BOARD_SIZE:
                    if self.state[1 - player][posX][posY]:
                        pass
                    elif self.state[player][posX][posY]:
                        if 1 < i:
                            for n in range(i):
                                self.state[player][n * dx + x][n * dy + y] = 1
                                self.state[1 - player][n * dx + x][n * dy + y] = 0
                            break
                        else:
                            break
                    else:
                        break
                else:
                    break

    def legalAction(self, x, y):
        player = self.getPlayer()

        for dx, dy in ((1, 1), (1, 0), (1, -1), (0, 1), (0, -1), (-1, 1), (-1, 0), (-1, -1)):
            for i in range(1, conf.BOARD_SIZE):
                posX, posY = i * dx + x, i * dy + y
                if 0 <= posX < conf.BOARD_SIZE and 0 <= posY < conf.BOARD_SIZE:
                    if self.state[1 - player][posX][posY]:
                        pass
                    elif self.state[player][posX][posY]:
                        if 1 < i:
                            return True
                        else:
                            break
                    else:
                        break
                else:
                    break

        return False

    def show(self):
        s = self.state[:2, :, :]
        for i in range(conf.BOARD_SIZE):
            for j in range(conf.BOARD_SIZE):
                if s[0, i, j]:
                    print("〇 ", end="")
                elif s[1, i, j]:
                    print("● ", end="")
                else:
                    print("・ ", end="")
            print("")


def randomAction(actions):
    n = random.randint(0, len(actions) - 1)
    return actions[n]
