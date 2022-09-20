import pickle
import logging
import numpy as np
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

        assert state_list.__len__() == action_list.__len__() == winner_list.__len__(), f'histry length error: ' \
                                                                                       f'{state_list.__len__()} - ' \
                                                                                       f'{action_list.__len__()} - ' \
                                                                                       f'{winner_list.__len__()}'
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
        train_history, valid_history, split_point = self._train_valid_split(rate=rate)
        return train_history, valid_history, split_point

    def _train_valid_split(self, rate=0.8):
        histry_length = self.state_list.__len__()
        split_point = int(histry_length * rate)

        train_history = PlayHistory(
            state_list=self.state_list[:split_point],
            action_list=self.action_list[:split_point],
            winner_list=self.winner_list[:split_point]
        )

        if rate == 1:
            valid_history = None
        else:
            valid_history = PlayHistory(
                state_list=self.state_list[split_point:],
                action_list=self.action_list[split_point:],
                winner_list=self.winner_list[split_point:]
            )

        return train_history, valid_history, split_point


def data_augment(play_history: PlayHistory, hflip=False, vflip=False, rot=False, max_action=-1):
    state_shape = play_history.state_list[0].shape
    if hflip:
        size = state_shape[1]

        def h_flip_action(action):
            if action == max_action:
                return action
            else:
                x = action // size
                y = action % size
                return x * size + ((size - 1) - y)
        
        h_flip_func = np.frompyfunc(h_flip_action, nin=1, nout=1)

        flip_state_array = np.stack(play_history.state_list, axis=0)
        flip_state_array = np.ascontiguousarray(flip_state_array[:, :, ::-1, :], dtype=flip_state_array.dtype)
        flip_state_list = list(flip_state_array)
        
        flip_action_array = np.array(play_history.action_list, dtype=np.int32)
        flip_action_array = h_flip_func(flip_action_array)
        flip_action_list = list(flip_action_array)
        
        flip_winner_list = [winner for winner in play_history.winner_list]
        del size

        play_history.append(state_list=flip_state_list, action_list=flip_action_list, winner_list=flip_winner_list)

    if vflip:
        size = state_shape[2]

        def v_flip_action(action):
            if action == max_action:
                return action
            else:
                x = action // size
                y = action % size
                return ((size - 1) - x) * size + y

        v_flip_func = np.frompyfunc(v_flip_action, nin=1, nout=1)
        
        flip_state_array = np.stack(play_history.state_list, axis=0)
        flip_state_array = np.ascontiguousarray(flip_state_array[:, :, :, ::-1], dtype=flip_state_array.dtype)
        flip_state_list = list(flip_state_array)

        flip_action_array = np.array(play_history.action_list, dtype=np.int32)
        flip_action_array = v_flip_func(flip_action_array)
        flip_action_list = list(flip_action_list)

        flip_winner_list = [winner for winner in play_history.winner_list]
        del size

        play_history.append(state_list=flip_state_list, action_list=flip_action_list, winner_list=flip_winner_list)

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
        return state, action, winner

    @staticmethod
    def collate_fn(batch):
        states, actions, winners = list(zip(*batch))
        states = torch.stack(states, dim=0)
        actions = torch.tensor(data=actions, dtype=torch.long)
        winners = torch.tensor(data=winners, dtype=torch.float)
        return states, actions, winners


def build_loader(self_play_histry, batch_size=64, num_workers=2, split_rate=0.8):
    train_dict, valid_dict, split_point = self_play_histry.get_train_valid_data(split_rate)
    train_data = AlphaDataset(train_dict)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=AlphaDataset.collate_fn,
        pin_memory=True
    )

    if valid_dict is not None:
        valid_data = AlphaDataset(valid_dict)
        valid_loader = DataLoader(
            dataset=valid_data,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=AlphaDataset.collate_fn,
            pin_memory=False
        )
        logging.info(msg=f'train_data size: {train_data.__len__()} - valid_data size: {valid_data.__len__()}')
    else:
        valid_loader = None
        logging.info(msg=f'train_data size: {train_data.__len__()} - valid_data size: 0')

    return train_loader, valid_loader
