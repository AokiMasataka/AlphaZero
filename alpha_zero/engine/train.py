import os
import sys
import copy
import logging
import warnings
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.nn import functional
from torch.cuda import amp

from .self_play import parallel_self_play
from ..games import GAMES
from ..model import ScaleModel
from ..utility import PlayHistory, AlphaDataset, AvgManager, data_augment


def model_train(model: ScaleModel, play_history: PlayHistory, train_config: dict, gen: int):
    device = train_config.get('device', 'cpu')
    model.to(device).train()
    train_history, valid_hostory = play_history.get_train_valid_data(train_data_rate=train_config['traindata_rate'])
    train_dataset = AlphaDataset(play_history=train_history)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=AlphaDataset.collate_fn,
    )
    if valid_hostory is not None:
        valid_dataset = AlphaDataset(play_history=valid_hostory)
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=train_config['batch_size'],
            shuffle=False,
            num_workers=4,
            collate_fn=AlphaDataset.collate_fn,
        )

    lr = train_config['base_lr'] * (train_config['lr_gamma'] ** gen)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    log_step = max(1, train_loader.__len__() // 4)

    scaler = amp.GradScaler(enabled=train_config.get('use_amp', False))
    
    for epoch in range(1, train_config['epochs'] + 1):
        train_loss = {'value': AvgManager(), 'policy': AvgManager()}
        for step, (state, action, winner) in enumerate(train_loader, 1):
            state, action, winner = state.to(device), action.to(device), winner.to(device)
            optimizer.zero_grad()

            with amp.autocast(enabled=train_config.get('use_amp', False)):
                value, policy = model(state)
            value_loss = functional.mse_loss(input=value.view(-1).float(), target=winner)
            policy_loss = functional.cross_entropy(input=policy.float(), target=action)
            loss = train_config['value_loss_weight'] * value_loss + train_config['policy_loss_weight'] * policy_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss['value'].update(value=value_loss.item())
            train_loss['policy'].update(value=policy_loss.item())

            if not(train_config.get('no_log', False)) and step % log_step == 0:
                msg = f'epochs: [{epoch}/{train_config["epochs"]}]'
                msg += f' - train value loss: {train_loss["value"]():.6f} - train policy loss: {train_loss["policy"]():.6f}'
                logging.info(msg=msg)

        if valid_loader is not None:
            valid_loss = {'value': AvgManager(), 'policy': AvgManager()}
            for states, actions, winners in valid_loader:
                states, actions, winners = states.to(device), actions.to(device), winners.to(device)
                with torch.no_grad():
                    value, policy = model(states)
                value_loss = functional.mse_loss(input=value.view(-1), target=winners)
                policy_loss = functional.cross_entropy(input=policy, target=actions)

                valid_loss['value'].update(value=value_loss.item())
                valid_loss['policy'].update(value=policy_loss.item())

        msg = f'epochs: [{epoch}/{train_config["epochs"]}]'
        msg += f' - train value loss: {train_loss["value"]():.6f} - train policy loss: {train_loss["policy"]():.6f}'

        if valid_loader is not None:
            msg += f' - valid value loss: {valid_loss["value"]():.6f} - valid policy loss: {valid_loss["policy"]():.6f}'
        logging.info(msg=msg)
    model.cpu().eval()
    return model


def model_evalate(new_model, old_model, evalate_config):
    num_evalate_play = evalate_config['num_games'] * 2

    _, f_winner = parallel_self_play(f_model=new_model, b_model=old_model, **evalate_config)
    _, b_winner = parallel_self_play(f_model=old_model, b_model=new_model, **evalate_config)
    b_winner = list(map(lambda x: -x, b_winner))
    
    # winarte = sum(f_winner + b_winner)
    winarte = (f_winner + b_winner).count(1)
    
    f_winrate = f_winner.count(1) / (num_evalate_play // 2)
    b_winrate = b_winner.count(1) / (num_evalate_play // 2)
    return winarte / num_evalate_play, f_winrate, b_winrate


def train(config: Dict[dict, dict, dict], save_play_history: bool = False):
    model_config = config['model_config']
    self_play_config = config['self_play_config']
    train_config = config['train_config']

    cuda_available = torch.cuda.is_available()
    logging.info(f'Python info: {sys.version}')
    logging.info(f'PyTroch version: {torch.__version__}')
    logging.info(f'CUDA available: {torch.cuda.is_available()}')
    if cuda_available:
        logging.info(f'GPU model: {torch.cuda.get_device_name(0)}')
    logging.info(msg=f'game name: {self_play_config["game"]} - init args: {self_play_config["init_dict"]}')

    if train_config.get('device', 'cpu') == 'cuda' and not cuda_available:
        warnings.warn(message='cuda is not available switch to cpu')
        train_config['device'] = 'cpu'
        train_config['use_amp'] = False


    game_module = GAMES.get_module(name=self_play_config.pop('game'))
    self_play_config['game_module'] = game_module
    game = game_module(**self_play_config['init_dict'])

    generation = self_play_config.pop('generation')

    if model_config.get('action_space', None) is None:
        model_config['action_space'] = game.action_space
    
    random_play_config = self_play_config.copy()
    random_play_config['random_play'] = 1000
    random_play_config['num_searchs'] = 1

    evalate_config = self_play_config.copy()
    evalate_config['num_games'] = 50 // 2
    
    model = ScaleModel(config=model_config)
    if model_config.get('pretrained', False):
        model.load_state_dict(torch.load(model_config['pretrained'], map_location='cpu'), strict=False)

    for gen in range(1, generation):
        if gen == 1:
            play_history, _ = parallel_self_play(f_model=model, b_model=model, **random_play_config)
        else:
            play_history, _ = parallel_self_play(f_model=model, b_model=model, **self_play_config)
        
        if train_config.get('aug', False):
            play_history = data_augment(
                play_history=play_history,
                hflip=train_config['hflip'],
                vflip=train_config['vflip'],
                max_action=game.action_space
            )
        old_model = copy.deepcopy(model)
        model = model_train(model=model, play_history=play_history, train_config=train_config, gen=gen)
        winrate, f_winrate, b_winrate = model_evalate(
            new_model=model, old_model=old_model, evalate_config=evalate_config
        )
        msg = f'new model winrate: {winrate:.4f}'
        msg += f' - fisrt hand winrate: {f_winrate:.4f}'
        msg += f' - back hand winrate: {b_winrate: 4f}\n'
        logging.info(msg=msg)

        save_dir = os.path.join(config['work_dir'], f'model_gen{gen}')
        model.save_pretrained(save_dir=save_dir, exist_ok=True)

        if save_play_history:
            play_history.save_history(save_path=os.path.join(save_dir, '/play_history.pkl'))
