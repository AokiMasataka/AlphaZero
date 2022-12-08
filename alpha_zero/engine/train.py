import torch
from torch.utils.data import Dataset, DataLoader
from ..self_play import PlayHistory


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


def tranner(train_config):