import torch
import numpy as np
import os
import sys

from model import ResNet
import self_play
import config as conf

conf.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Main:
    def __init__(self, loadModelGen=None):
        os.makedirs('model/' + conf.PATH, exist_ok=True)
        os.makedirs('data/' + conf.PATH, exist_ok=True)
        self.Model = ResNet().to(conf.DEVICE)
        if loadModelGen == None:
            self.modelGen = 0
            print("modelGen : ", self.modelGen)
            data = self_play.randomData()
            data = self_play.inflated(data)
            self.Model.fit(data, policyVias=1, valueVias=1)
            np.savez('data/' + conf.PATH + '/Gen' + str(self.modelGen), data[0], data[1], data[2])
            torch.save(self.Model.state_dict(), 'model/' + conf.PATH + '/Gen' + str(self.modelGen))
        else:
            self.modelGen = loadModelGen
            self.Model.load_state_dict(torch.load('model/' + conf.PATH + '/Gen' + str(self.modelGen)))

    def train(self):
        while True:
            self.modelGen += 1
            if self.modelGen == 11:
                break
            print("modelGen : ", self.modelGen)
            data = self_play.DataGenerate(self.Model)
            data = self_play.inflated(data)
            self.Model.fit(data, policyVias=1, valueVias=1)
            np.savez('data/' + conf.PATH + '/Gen' + str(self.modelGen), data[0], data[1], data[2])
            torch.save(self.Model.state_dict(), 'model/' + conf.PATH + '/Gen' + str(self.modelGen))


if __name__ == '__main__':
    args = sys.argv
    length = len(args) - 1
    if length:
        main = Main(int(args[1]))
    else:
        main = Main()
    main.train()
