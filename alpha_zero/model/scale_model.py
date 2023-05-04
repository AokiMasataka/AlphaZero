import os
import json
import torch
from torch import nn
from ..games import BaseGame
from ..utility import load_config


class Block(nn.Module):
    def __init__(self, dim, eps, momentum, se=False, act_fn='relu'):
        super(Block, self).__init__()
        if act_fn == 'relu':
            act_fn = nn.ReLU
        elif act_fn == 'gelu':
            act_fn = nn.GELU
        else:
            raise 'act_fn is relu or gelu'
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=dim, eps=eps, momentum=momentum)
        self.act1 = act_fn()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=dim, eps=eps, momentum=momentum)
        self.act2 = act_fn()

        self.se = se
        if se:
            self.se_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(start_dim=1, end_dim=-1),
                nn.Linear(in_features=dim, out_features=dim // 4),
                act_fn(),
                nn.Linear(in_features=dim // 4, out_features=dim),
                nn.Sigmoid()
            )
        self.dim = dim

    def forward(self, x):
        skip = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.se:
            x = self.se_layer(x).view(-1, self.dim, 1, 1) * x
        return self.act2(x + skip)


class ScaleModel(nn.Module):
    def __init__(self, config: dict):
        super(ScaleModel, self).__init__()
        assert 'stem_config' in config.keys(), 'not stem_config in config'
        assert 'block_config' in config.keys(), 'not block_config in config'
        self.config = config
        stem_config = config['stem_config']
        block_config = config['block_config']

        self.stem = nn.Conv2d(
            in_channels=stem_config['in_channels'],
            out_channels=stem_config['out_dim'],
            kernel_size=stem_config['kernel_size'],
            stride=stem_config['stride'],
            padding=stem_config['padding']
        )
        self.blocks = nn.Sequential(*[Block(**block_config) for _ in range(config['depth'])])

        self.value_head = nn.Sequential(
            nn.Conv2d(block_config['dim'], block_config['dim'], kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(block_config['dim'], 1),
            nn.Tanh()
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(block_config['dim'], block_config['dim'], kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(block_config['dim'], config['action_space']),
            nn.Softmax(dim=1)
        )
    
    def extract_feature(self, x):
        x = self.stem(x)
        return self.blocks(x)

    def forward(self, x):
        feature = self.extract_feature(x)
        value = self.value_head(feature)
        policy = self.policy_head(feature)
        return value, policy
        
    @torch.inference_mode()
    def get_value(self, obj: BaseGame):
        inputs = torch.tensor(obj.encode_state(), dtype=torch.float).unsqueeze(0)
        feature = self.extract_feature(inputs)
        return self.value_head(feature).cpu().item()

    @torch.inference_mode()
    def get_policy(self, obj: BaseGame):
        inputs = torch.tensor(obj.encode_state(), dtype=torch.float).unsqueeze(0)
        feature = self.extract_feature(inputs)
        return self.policy_head(feature).cpu().squeeze(0).numpy()

    @torch.inference_mode()
    def inference_state(self, obj: BaseGame):
        inputs = torch.tensor(obj.encode_state(), dtype=torch.float).unsqueeze(0)
        feature = self.extract_feature(inputs)
        value = self.value_head(feature)
        policy = self.policy_head(feature)
        return value.cpu().item(), policy.cpu().squeeze(0).numpy()

    def save_pretrained(self, save_dir, exist_ok=False):
        if exist_ok:
            os.makedirs(save_dir, exist_ok=True)

        with open(os.path.join(save_dir, 'model_config.json'), mode='w') as f:
            json.dump(obj=self.config, fp=f, indent=4)
        
        torch.save(self.state_dict(), save_dir + '/model.pth')

    @staticmethod
    def from_pretrained(load_dir):

        with open(os.path.join(load_dir, 'model_config.json'), mode='r') as f:
            config = json.load(fp=f)
        model = ScaleModel(config)
        model.load_state_dict(torch.load(load_dir + '/model.pth'))
        return model

    @staticmethod
    def from_config(config):
        if isinstance(config, str):
            config, _ = load_config(config)
        elif isinstance(config, dict):
            pass
        
        return ScaleModel(config=config['model_config'])
