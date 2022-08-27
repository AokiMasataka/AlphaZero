import pickle
import torch
from torch.utils.data import Dataset, DataLoader


class PlayHistory:
    def __init__(self, state_list=None, action_list=None, winner_list=None):
        if state_list is not None:
            self.state_list = state_list
            self.action_list = action_list
            self.winner_list = winner_list
        else:
            self.state_list = list()
            self.action_list = list()
            self.winner_list = list()

    def __call__(self):
        pass

    def __len__(self):
        return self.state_list.__len__()

    def __add__(self, other):
        self.state_list += other.state_list
        self.action_list += other.action_list
        self.winner_list += other.winner_list
        return self

    def __getitem__(self, index, return_dict=False):
        if return_dict:
            return {
                'state': self.state_list[index],
                'action': self.action_list[index],
                'winnner': self.winner_list[index]
            }
        else:
            return self.state_list[index], self.action_list[index], self.winner_list[index]

    def append(self, state_list=None, action_list=None, winner_list=None, history=None):
        if history is not None:
            assert isinstance(history, PlayHistory), 'history class'
            history.data_check()
            self.state_list += history.state_list
            self.action_list += history.action_list
            self.winner_list += history.winner_list

        assert state_list.__len__() == action_list.__len__() == winner_list.__len__(), 'histry length error'
        self.state_list += state_list
        self.action_list += action_list
        self.winner_list += winner_list

    def data_check(self):
        assert len(self.state_list) == len(self.action_list) == len(self.winner_list), 'histry length error'

    def save_histry(self, save_path):
        histry_dict = {
            'state_list': self.state_list,
            'action_list': self.action_list,
            'winner_list': self.winner_list
        }

        with open(save_path, 'wb') as f:
            pickle.dump(histry_dict, f)

    def load_histry(self, laod_path):
        with open(laod_path, 'rb') as f:
            histry_dict = pickle.load(f)

        self.state_list = histry_dict['state_list']
        self.action_list = histry_dict['action_list']
        self.winner_list = histry_dict['winner_list']

    def get_train_valid_data(self, rate=0.8):
        train_dict, valid_dict, split_point = self._train_valid_split(rate=rate)
        return train_dict, valid_dict, split_point

    def _train_valid_split(self, rate=0.8):
        histry_length = self.state_list.__len__()
        split_point = int(histry_length * rate)

        train_history = PlayHistory(
            state_list=self.state_list[:split_point],
            action_list=self.action_list[:split_point],
            winner_list=self.winner_list[:split_point]
        )

        valid_history = PlayHistory(
            state_list=self.state_list[split_point:],
            action_list=self.action_list[split_point:],
            winner_list=self.winner_list[split_point:]
        )

        return train_history, valid_history, split_point


def data_augment(play_history: PlayHistory, hflip=False, vflip=False, rot=False):
    if hflip:
        pass

    if vflip:
        pass

    if rot:
        pass

    return play_history


class AlphaDataset(Dataset):
    def __init__(self, play_histry: PlayHistory):
        play_histry.data_check()

        self.state_list = play_histry.state_list
        self.action_list = play_histry.action_list
        self.winner_list = play_histry.winner_list

    def __len__(self):
        return self.state_list.__len__()

    def __getitem__(self, index):
        state = self.state_list[index]
        action = self.action_list[index]
        winner = self.winner_list[index]

        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        winner = torch.tensor(winner, dtype=torch.float)
        return state, action, winner


def build_loader(self_play_histry, batch_size=64, split_rate=0.8):
    train_dict, valid_dict, split_point = self_play_histry.get_train_valid_data(split_rate)
    print(f'INFO: train data size: {split_point}')
    train_data = AlphaDataset(train_dict)
    valid_data = AlphaDataset(valid_dict)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, num_workers=4)
    return train_loader, valid_loader
