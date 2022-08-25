import os
import json
import torch
from torch import nn


class ModelConfig:
    def __init__(self, in_channels=2, dim=256, depth=4, max_actions=1, eps=1e-6, momentum=0.1):
        self.in_channels = in_channels
        self.dim = dim
        self.depth = depth
        self.max_actions = max_actions
        self.eps = eps
        self.momentum = momentum
        self.act_fn = 'relu'

    def to_dict(self):
        return self.__dict__

    def save_config(self, save_dir):
        with open(save_dir + '/config.json', 'w') as f:
            json.dump(self.__dict__, f)

    @staticmethod
    def load_config(load_dir):
        with open(load_dir + '/config.json', 'r') as f:
            config_dict = json.load(f)
        return ModelConfig(**config_dict)


class Block(nn.Module):
    def __init__(self, dim, eps, momentum, act_fn='relu'):
        super(Block, self).__init__()
        assert act_fn in ('relu', 'gelu')
        if act_fn == 'relu':
            act_fn = nn.ReLU
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=dim, eps=eps, momentum=momentum)
        self.act1 = act_fn()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=dim, eps=eps, momentum=momentum)
        self.act2 = act_fn()

    def forward(self, x):
        cat = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        return x + cat


class ScaleModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(ScaleModel, self).__init__()
        self.config = config
        self.stem = nn.Conv2d(config.in_channels, out_channels=config.dim, kernel_size=(2, 2), stride=(2, 2), padding=0)
        block_args = {'dim': config.dim, 'eps': config.eps, 'momentum': config.momentum, 'act_fn': config.act_fn}
        self.blocks = nn.Sequential(*[Block(**block_args) for _ in range(config.depth)])

        self.value_head = nn.Sequential(
            nn.Conv2d(config.dim, config.dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(config.dim, 1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(config.dim, config.dim, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(config.dim, config.max_actions),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feature = self.extract_feature(x)
        value = self.value_head(feature)
        policy = self.policy_head(feature)
        return value, policy

    def get_value(self, x):
        feature = self.extract_feature(x)
        return self.value_head(feature)

    def get_policy(self, x):
        feature = self.extract_feature(x)
        return self.policy_head(feature)

    def extract_feature(self, x):
        x = self.stem(x)
        return self.blocks(x)

    @torch.inference_mode
    def inference_state(self, state):
        x = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        feature = self.extract_feature(x)
        value = self.value_head(feature)
        policy = self.policy_head(feature)
        return value, policy

    def save_pretrained(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        self.config.save_config(save_dir)
        torch.save(self.state_dict(), save_dir)

    @staticmethod
    def from_pretrained(laod_dir):
        config = ModelConfig.load_config(laod_dir)
        model = ScaleModel(config)
        model.laod_state_dict(torch.load(laod_dir + 'model.pth'))
        return model
