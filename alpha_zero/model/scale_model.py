import json
import torch
from torch import nn, Tensor
from ..games import Game
# from ..utils import load_config


class Block(nn.Module):
    def __init__(self, embed_dim: int, se_layer: bool = False, rd_ratio: int = 8, eps: float = 1e-5) -> None:
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.norm1 = nn.BatchNorm2d(num_features=embed_dim, eps=eps)
        self.activation1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )
        self.norm2 = nn.BatchNorm2d(num_features=embed_dim, eps=eps)
        self.activation2 = nn.ReLU(inplace=True)

        if se_layer:
            self.se_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(embed_dim, embed_dim // rd_ratio, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                nn.SiLU(),
                nn.Conv2d(embed_dim // rd_ratio, embed_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
                nn.Sigmoid()
            )
        else:
            self.se_layer = None
    
    def forward(self, x: Tensor) -> Tensor:
        cat = x
        x = self.activation1(self.norm1(self.conv1(x)))
        x = self.activation2(self.norm2(self.conv2(x)))
        
        if self.se_layer is not None:
            x *= self.se_layer(x)

        return x + cat


class ScaleModel(nn.Module):
    def __init__(
        self,
        stem: dict = dict(
            in_channels=2,
            embed_dim=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        ),
        block: dict = dict(
            embed_dim=128,
            se_layer=False,
        ),
        num_layers: int = 8,
        action_space: int = None,
        eps: float = 1e-5,
        train_cache: bool = False
    ):
        super(ScaleModel, self).__init__()
        self.config = {
            'stem': stem,
            'block': block,
            'num_layers': num_layers,
            'action_space': action_space,
            'eps': eps,
            'train_cache': train_cache
        }

        self.embed_dim = block['embed_dim']
        stem['out_channels'] = stem.pop('embed_dim')
        self.stem = nn.Sequential(
            nn.Conv2d(**stem),
            nn.BatchNorm2d(num_features=self.embed_dim, eps=eps),
            nn.ReLU(inplace=True)
        )

        self.blocks = nn.Sequential()
        for _ in range(num_layers):
            self.blocks.append(Block(**block))
        
        self.policy = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim * 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.embed_dim * 2, eps=eps),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_features=self.embed_dim * 2, out_features=action_space),
            nn.Softmax(dim=1)
        )

        self.value = nn.Sequential(
            nn.Conv2d(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.embed_dim, eps=eps),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_features=self.embed_dim, out_features=1),
            nn.Tanh()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_feature(x=x)
        policy = self.policy(x)
        value = self.value(x)
        return value, policy

    def forward_inference(self, game: Game) -> dict:
        x = torch.tensor(game.encode_state(), dtype=torch.float)
        x.unsqueeze_()
        x = self.forward_feature(x=x)
        polciy = self.policy(x)
        value = self.value(x)
        return {'policy': polciy, 'value': value}

    def forward_feature(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return x
    
    def forward_policy(self, x: Tensor) -> Tensor:
        x = self.forward_feature(x=x)
        policy = self.policy(x)
        return policy
    
    def forward_value(self, x: Tensor) -> Tensor:
        x = self.forward_feature(x=x)
        value = self.value(x)
        return value
    
    def save_pretrained(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.config, fp=f, indent=4)
    
    @staticmethod
    def load_pretrained(self, path: str):
        with open(path, 'r') as f:
            config = json.load(f)
        
        model = ScaleModel(**config)
        return model
    
    @staticmethod
    def load_from_config(config: dict):
        model = ScaleModel(**config['model'])
        return model