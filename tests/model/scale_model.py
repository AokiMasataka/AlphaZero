import os
import sys
sys.path.append(os.getcwd())

import torch
from copy import deepcopy
from alpha_zero.model import ScaleModel


def main():
    model_config = dict(
        stem=dict(
            in_channels=2,
            embed_dim=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        ),
        block=dict(
            embed_dim=128,
            se_layer=True,
            rd_ratio=8,
        ),
        num_layers=8,
        action_space=64,
        eps=1e-5,
        train_cache=False
    )
    model = ScaleModel(**deepcopy(model_config))
    model = ScaleModel.load_from_config(config={'model': deepcopy(model_config)})

    x = torch.rand(1, 2, 8, 8, dtype=torch.float)
    value, policy = model(x=x)

    assert value.shape == (1, 1)
    assert policy.shape == (1, 64)
    print('Successfully')


if __name__ == '__main__':
    main()
