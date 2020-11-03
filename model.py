import torch
from torch import nn

import config as conf


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Block(nn.Module):
    def __init__(self, Dim):
        super(Block, self).__init__()
        Dim = int(Dim / 4)
        self.conv1 = nn.Conv2d(Dim * 4, Dim, kernel_size=1)
        self.conv2 = nn.Conv2d(Dim, Dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(Dim, Dim * 4, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(Dim)
        self.bn2 = nn.BatchNorm2d(Dim)
        self.bn3 = nn.BatchNorm2d(Dim * 4)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        y = self.relu3(x + inputs)
        return y


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, conf.FILTERS, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([Block(conf.FILTERS) for _ in range(conf.BLOCKS)])

        self.policyHead = nn.Sequential(
            nn.Conv2d(256, 8, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            Flatten(),
            nn.Linear(8 * conf.MAXIMUM_ACTION, conf.MAXIMUM_ACTION + 1),
            nn.Softmax(dim=1)
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(256, 4, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            Flatten(),
            nn.Linear(4 * conf.MAXIMUM_ACTION, 64),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv1(x)

        for block in self.blocks:
            x = block(x)

        value = self.valueHead(x)
        policy = self.policyHead(x)

        return policy, value

    def fit(self, data, policyVias=1, valueVias=1):
        state, policy, value = data[0], data[1], data[2]

        tensor_state = torch.tensor(state, dtype=torch.float).to(conf.DEVICE)
        tensor_policy = torch.tensor(policy, dtype=torch.long).to(conf.DEVICE)
        tensor_value = torch.tensor(value, dtype=torch.float).to(conf.DEVICE)

        train_state = tensor_state[conf.TEST_DATA_NUM:]
        train_policy = tensor_policy[conf.TEST_DATA_NUM:]
        train_value = tensor_value[conf.TEST_DATA_NUM:]
        test_state = tensor_state[:conf.TEST_DATA_NUM]
        test_policy = tensor_policy[:conf.TEST_DATA_NUM]
        test_value = tensor_value[:conf.TEST_DATA_NUM]
        print("train Data", train_state.shape, train_policy.shape, train_value.shape)
        print("test Data", test_state.shape, test_policy.shape, test_value.shape)

        trainDS = (train_state, train_policy, train_value)
        trainLoader = DataLoader(trainDS, batch_size=conf.BATCH_SIZE, shuffle=True)
        testDS = (test_state, test_policy, test_value)
        testLoader = DataLoader(testDS, batch_size=100, shuffle=False)

        optimizer = torch.optim.SGD(self.parameters(), lr=conf.LEARNING_RATE, momentum=conf.MOMENTUM)
        policyLossF = nn.CrossEntropyLoss()
        valueLossF = nn.MSELoss()

        iterate = int(len(state) / conf.BATCH_SIZE)
        for epoch in range(conf.EPOCHS):
            losses = [0, 0, 0]
            accuracy = [0, 0]
            self.train()
            for i, (X, Y_policy, Y_value) in enumerate(trainLoader):
                p, v = self.forward(X)

                optimizer.zero_grad()
                p_loss = policyLossF(p, Y_policy)
                v_loss = valueLossF(v.view(-1), Y_value)

                loss = (p_loss * policyVias) + (v_loss * valueVias)
                loss.backward()
                optimizer.step()

                losses[0] += loss.item()
                losses[1] += p_loss.item()
                losses[2] += v_loss.item()

            self.eval()
            for i, (X, Y_policy, Y_value) in enumerate(testLoader):
                p, v = self.forward(X)
                _, pred = torch.max(p, dim=1)
                accuracy[0] += pred.eq(Y_policy).sum().item() / Y_value.shape[0]
                accuracy[1] += (v.view(-1).round() == Y_value).sum().item() / Y_value.shape[0]

            print('total Loss:', losses[0] / iterate)
            print('policy Loss:', losses[1] / iterate, ' policy Accuracy', (100.0 * accuracy[0] / conf.TEST_DATA_NUM))
            print('value Loss:', losses[2] / iterate, ' value Accuracy', 100.0 * accuracy[1] / conf.TEST_DATA_NUM)
            print("")


class DataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert all([dataset[i].shape[0] == dataset[0].shape[0] for i in range(len(dataset))]), 'all the elemtnes must have the same length'
        self.data_size = dataset[0].shape[0]

    def __iter__(self):
        self._i = 0
        if self.shuffle:
            index_shuffle = torch.randperm(self.data_size)
            self.dataset = [v[index_shuffle] for v in self.dataset]
        return self

    def __next__(self):
        i1 = self.batch_size * self._i
        i2 = min(self.batch_size * (self._i + 1), self.data_size)
        if i1 >= self.data_size:
            raise StopIteration()
        value = [v[i1:i2] for v in self.dataset]
        self._i += 1
        return value
