import numpy as np
from ctypes import *
import random
import torch
from tqdm import tqdm

from game import Game
import config as conf


class Data:
    def __init__(self):
        self.game = Game()
        self.states = np.empty((conf.MAXIMUM_ACTION + 16, 3, conf.BOARD_SIZE, conf.BOARD_SIZE), dtype=np.int)
        self.actions = np.empty([conf.MAXIMUM_ACTION + 16], dtype=np.int)
        self.Q = False
        self.isDone = 0
        self.actList = []
        self.gameCount = 0

    def randomPlay(self, num):
        for i in range(num):
            if self.game.isDone():
                self.isDone = 1
                break
            actions = self.game.getLegalAction()
            self.states[self.gameCount] = self.game.state
            if len(actions):
                action = actions[np.random.randint(0, len(actions))]

                self.actions[self.gameCount] = action
                self.game.action(action)
            else:
                self.actions[self.gameCount] = conf.MAXIMUM_ACTION
                self.game.changePlayer()
            self.gameCount += 1

    def play(self):
        if self.game.isDone():
            self.isDone = 1
        self.actList = self.game.getLegalAction()
        self.states[self.gameCount] = self.game.state
        if 1 < len(self.actList):
            self.Q = True
        elif 1 == len(self.actList):
            action = self.actList[0]
            self.actions[self.gameCount] = action
            self.game.action(action)
            self.Q = False
        else:
            self.actions[self.gameCount] = conf.MAXIMUM_ACTION
            self.game.changePlayer()
            self.Q = False
        self.gameCount += 1

    def MCTS_action(self, action):
        try:
            action = self.actList[action]
        except:
            action = self.actList[-1]
        self.actions[self.gameCount - 1] = action
        self.game.action(action)

    def getState(self):
        return self.states[:self.gameCount]

    def getActions(self):
        return self.actions[:self.gameCount]

    def getWinner(self):
        winner = self.game.getWinner()
        return np.full(self.gameCount, winner)


def DataGenerate(model):
    model.eval()
    MCTS = cdll.LoadLibrary("monte_carlo_" + conf.PATH + ".dll")
    MCTS.setC.argtypes = [c_float]
    MCTS.clear.argtypes = []
    MCTS.init.argtypes = [POINTER(c_int)]
    MCTS.moveToLeaf.argtypes = [c_int]
    MCTS.riseNode.argtypes = [POINTER(c_float), POINTER(c_float), c_int]
    MCTS.getAction.argtypes = [c_float, c_int]
    MCTS.getState.argtypes = [POINTER(c_int), c_int]

    datas = [Data() for i in range(conf.GENERATE_NUM)]

    for game in datas:
        game.randomPlay(conf.RANDOM_CHOICE)
    MCTS.setC(conf.C)
    while True:
        done = 0
        count = 0
        for data in datas:
            data.play()
            if data.Q:
                s = data.game.state.reshape(-1).astype(np.int)
                a = np.ctypeslib.as_ctypes(s)
                MCTS.init(a)
                count += 1
            elif data.isDone:
                done += data.isDone
        if done == conf.GENERATE_NUM:
            break

        if not count:
            continue

        get = np.zeros(3 * conf.MAXIMUM_ACTION * count).astype(np.int)
        c_get = np.ctypeslib.as_ctypes(get)
        for i in tqdm(range(conf.SEARCH_NUM)):
            MCTS.moveToLeaf(count)

            MCTS.getState(c_get, count)
            get = np.ctypeslib.as_array(c_get)
            X = get.reshape(-1, 3, conf.BOARD_SIZE, conf.BOARD_SIZE)

            X = torch.tensor(X, dtype=torch.float).to(conf.DEVICE)
            policy, value = model(X)
            policy = policy.cpu().detach().numpy().reshape(-1)
            value = value.cpu().detach().numpy().reshape(-1)

            c_policy = np.ctypeslib.as_ctypes(policy)
            c_value = np.ctypeslib.as_ctypes(value)
            MCTS.riseNode(c_value, c_policy, count)

        index = 0
        for i in range(conf.GENERATE_NUM):
            if datas[i].Q:
                action = MCTS.getAction(-1, index)
                datas[i].MCTS_action(action)
                index += 1
        MCTS.clear()

    states = np.empty((1, 3, conf.BOARD_SIZE, conf.BOARD_SIZE))
    actions = np.array([])
    winners = np.array([])
    for data in datas:
        states = np.concatenate((states, data.getState()), axis=0)
        actions = np.append(actions, data.getActions())
        winners = np.append(winners, data.getWinner())
    states = states[1:]
    return [states, actions, winners]


def randomData():
    def randomPlay():
        game = Game()
        states = np.empty((conf.MAXIMUM_ACTION + 4, 3, conf.BOARD_SIZE, conf.BOARD_SIZE), dtype=np.int)
        actions = np.empty([conf.MAXIMUM_ACTION + 4], dtype=np.int)
        gameCount = 0

        game.__init__()
        while True:
            if game.isDone():
                break
            legalAction = game.getLegalAction()
            states[gameCount] = game.state
            if len(legalAction):
                n = random.randint(0, len(legalAction) - 1)
                game.action(legalAction[n])
                actions[gameCount] = legalAction[n]
            else:
                game.changePlayer()
                actions[gameCount] = conf.MAXIMUM_ACTION
            gameCount += 1

        winner = np.full(gameCount, game.getWinner())
        return states[:gameCount], actions[:gameCount], winner

    states = np.empty((1, 3, conf.BOARD_SIZE, conf.BOARD_SIZE))
    actions = np.array([])
    winners = np.array([])

    for i in tqdm(range(conf.RANDOM_PLAY_NUM)):
        state, action, winner = randomPlay()
        states = np.concatenate((states, state), axis=0)
        actions = np.append(actions, action)
        winners = np.append(winners, winner)
    states = states[1:]
    return [states, actions, winners]


def inflated(data):
    state, actions, win = data[0], data[1], data[2]

    for i in range(2):
        state = np.concatenate((state, np.rot90(state, i + 1, axes=(3, 2))), axis=0)
        win = np.append(win, win)

    rotActions = (actions / conf.BOARD_SIZE).astype(np.int) + (
                conf.BOARD_SIZE - 1 - (actions % conf.BOARD_SIZE).astype(np.int)) * conf.BOARD_SIZE
    actions = np.concatenate((actions, rotActions), axis=0)
    rotActions = (actions / conf.BOARD_SIZE).astype(np.int) + (
                conf.BOARD_SIZE - 1 - (actions % conf.BOARD_SIZE).astype(np.int)) * conf.BOARD_SIZE
    rotActions = (rotActions / conf.BOARD_SIZE).astype(np.int) + (
                conf.BOARD_SIZE - 1 - (rotActions % conf.BOARD_SIZE).astype(np.int)) * conf.BOARD_SIZE
    actions = np.concatenate((actions, rotActions), axis=0)

    state = np.concatenate((state, np.flip(state, 3)), axis=0)
    win = np.append(win, win)
    actions = np.append(actions, (actions % conf.BOARD_SIZE) + (
                conf.BOARD_SIZE - 1 - (actions / conf.BOARD_SIZE).astype(np.int)) * conf.BOARD_SIZE)

    for i in range(len(actions)):
        if actions[i] == -conf.BOARD_SIZE:
            actions[i] = conf.MAXIMUM_ACTION
    return [state, actions, win]
