import os
import copy
import logging
import torch
from torch.nn import functional
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
from models import ScaleModel
from ..self_play import PlayHistory, parallel_self_play
from ..games import GAMES
from ..utils import AvgManager


class AlphaDataset(Dataset):
    def __init__(self, play_history: PlayHistory):
        play_history.data_check()
        self.state_list = play_history.state_list
        self.action_list = play_history.action_list
        self.winner_list = play_history.winner_list

    def __len__(self):
        return self.state_list.__len__()

    def __getitem__(self, index):
        state = self.state_list[index]
        action = self.action_list[index]
        winner = self.winner_list[index]

        state = torch.tensor(state, dtype=torch.float)
        return state, action, winner
    
    @staticmethod
    def collate_fn(batch):
        states, actions, winners = list(zip(*batch))
        states = torch.stack(states, dim=0)
        actions = torch.tensor(data=actions, dtype=torch.long)
        winners = torch.tensor(data=winners, dtype=torch.float)
        return states, actions, winners


def model_train(model, play_history: PlayHistory, train_config: dict, gen: int):
    device = train_config.get('devica', 'cpu')
    model.train()
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
    optimizer = torch.optim.adam(model.parameters(), lr=lr)
    log_step = train_loader.__len__() // 10

    scaler = amp.GradScaler(enabled=train_config.get('use_amp', False))
    
    for epoch in range(1, train_config['epochs'] + 1):
        train_value_mean = AvgManager()
        train_policy_mean = AvgManager()
        for step, (state, action, winner) in enumerate(train_loader, 1):
            state, action, winner = state.to(device), action.to(device), winner.to(device)
            optimizer.zero_grad()

            with amp.autocast(enabled=train_config.get('use_amp', False)):
                value, policy = model(state)
            
            value_loss = functional.mse_loss(input=value.view(-1), target=winner)
            policy_loss = functional.cross_entropy(input=policy, target=action)
            loss = train_config['value_loss_weight'] * value_loss + train_config['policy_loss_weight'] * policy_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_value_mean.update(value=value_loss.item())
            train_policy_mean.update(value=policy_loss.item())

            if not(train_config.get('no_log', False)) and step % log_step == 0:
                msg = f'epochs: [{epoch}/{train_config["epochs"]}]'
                msg += f' - train value loss: {train_value_mean():.6f} - train policy loss: {train_policy_mean():.6f}'
                logging.info(msg=msg)
        
        if valid_loader is not None:
            valid_value_mean = AvgManager()
            valid_policy_mean = AvgManager()
            for states, actions, winners in valid_loader:
                states, actions, winners = states.to(device), actions.to(device), winners.to(device)
                with torch.no_grad():
                    value, policy = model(states)
                    value_loss = functional.mse_loss(input=value.view(-1), target=winners)
                    policy_loss = functional.cross_entropy(input=policy, target=actions)

                value_loss = value_loss.item()
                policy_loss = policy_loss.item()

                valid_value_mean.update(value=value_loss)
                valid_policy_mean.update(value=policy_loss)
        
        msg = f'epochs: [{epoch}/{train_config["epochs"]}]'
        msg += f' - train value loss: {train_value_mean():.6f} - train policy loss: {train_policy_mean():.6f}'

        if valid_loader is not None:
            msg += f' - valid value loss: {valid_value_mean():.6f} - valid policy loss: {valid_policy_mean():.6f}'
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


def train(config: dict, save_play_history: bool):
    model_config = config['model_config']
    self_play_config = config['self_play_config']
    train_config = config['train_config']

    game_module = GAMES.get_module(name=self_play_config['game'])
    game = game_module(**self_play_config['init_dict'])

    logging.info(msg=f'game name: {self_play_config["game"]} - init args: {self_play_config["init_dict"]}')

    if model_config.get('action_space', None) is None:
        model_config['action_space'] = game.action_space
    
    random_play_config = copy.deepcopy(self_play_config)
    random_play_config['random_play'] = 1000
    random_play_config['num_searchs'] = 1

    evalate_config = copy.deepcopy(self_play_config)
    evalate_config['num_games'] = 50 // 2
    
    model = ScaleModel(config=model_config)

    for gen in range(1, self_play_config['generation']):
        if gen == 1:
            play_history, _ = parallel_self_play(f_model=model, b_model=model, **random_play_config)
        else:
            play_history, _ = parallel_self_play(f_model=model, b_model=model, **self_play_config)
        
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
            play_history.save_histry(save_path=os.path.join(save_dir, '/play_history.pkl'))
