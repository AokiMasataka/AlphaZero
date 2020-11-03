import torch
import numpy as np
import tkinter as tk
from tqdm import tqdm
from ctypes import *
import sys

from model import ResNet
import config as conf
from game import Game

size = 600
square = size / conf.BOARD_SIZE
half = square / 2


class Play:
    def __init__(self, modelGen):
        self.game = Game()
        self.model = ResNet()
        self.model.load_state_dict(torch.load('model/' + conf.PATH + '/Gen' + str(modelGen)))
        self.model.eval().cpu()

        self.MCTS = cdll.LoadLibrary("monte_carlo_" + conf.PATH + ".dll")
        self.MCTS.setC.argtypes = [c_float]
        self.MCTS.clear.argtypes = []
        self.MCTS.SingleInit.argtypes = [POINTER(c_int)]
        self.MCTS.SingleMoveToLeaf.argtypes = []
        self.MCTS.SingleRiseNode.argtypes = [c_float, POINTER(c_float)]
        self.MCTS.SingleGetAction.argtypes = [c_float]
        self.MCTS.SingleGetState.argtypes = [POINTER(c_int)]

        print("W :", self.getValue())

    def draw(self):
        for i in range(conf.BOARD_SIZE):
            for j in range(conf.BOARD_SIZE):
                canvas.create_rectangle(i * square, j * square, i * square + square, j * square + square, fill='green', outline='black')
                if self.game.state[0][j][i]:
                    canvas.create_oval(i * square + 2, j * square + 2, i * square + square - 2, j * square + square - 2, fill='black')
                elif self.game.state[1][j][i]:
                    canvas.create_oval(i * square + 2, j * square + 2, i * square + square - 2, j * square + square - 2, fill='white')

    def monteCarlo(self):
        state = self.game.state.reshape(-1).astype(np.int)
        c_state = np.ctypeslib.as_ctypes(state)
        self.MCTS.SingleInit(c_state)

        get = np.zeros(3 * conf.MAXIMUM_ACTION).astype(np.int32)
        c_get = np.ctypeslib.as_ctypes(get)
        for i in tqdm(range(conf.SEARCH_NUM)):
            self.MCTS.SingleMoveToLeaf()

            self.MCTS.SingleGetState(c_get)
            get = np.ctypeslib.as_array(c_get)
            X = get.reshape(1, 3, conf.BOARD_SIZE, conf.BOARD_SIZE)

            policy, value = self.model(torch.tensor(X, dtype=torch.float))

            policy = policy.detach().numpy().reshape(-1)
            value = value.detach().numpy().reshape(-1)
            c_policy = np.ctypeslib.as_ctypes(policy)
            c_value = np.ctypeslib.as_ctypes(value)[0]

            self.MCTS.SingleRiseNode(c_value, c_policy)

        action = self.MCTS.SingleGetAction(conf.TEMP)
        return action

    def getValue(self):
        policy, value = self.model(torch.from_numpy(self.game.state).unsqueeze(0).float())
        return value.detach().numpy()[0][0]

    def click(self, event):
        x = int(event.x / square)
        y = int(event.y / square)
        action = x * conf.BOARD_SIZE + y

        if action in self.game.getLegalAction():
            self.game.action(action)
        else:
            self.game.changePlayer()

        actions = self.game.getLegalAction()
        if len(actions):
            action = self.monteCarlo()
            self.game.action(actions[action])
            print("W :", self.getValue())
        else:
            self.game.changePlayer()
        print("black:", int(np.sum(self.game.state[0, :, :])), " white:", int(np.sum(self.game.state[1, :, :])))

        self.draw()


if __name__ == '__main__':
    args = sys.argv
    play = Play(int(args[1]))

    root = tk.Tk()
    root.title(u"Othello")
    root.geometry("800x600")
    canvas = tk.Canvas(root, width=size, height=size)
    canvas.place(x=-1, y=-1)

    play.draw()
    canvas.bind("<Button-1>", play.click)

    root.mainloop()