import logging
import numpy as np
import torch
from torch.utils.data import Dataset
from .structs import PlayHistory


class AlphaDataset(Dataset):
    def __init__(self, play_history: PlayHistory):
        play_history._data_check()
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


def data_augment(play_history: PlayHistory, hflip: bool = False, vflip: bool = False, max_action: int = -1):
    state_shape = play_history.state_list[0].shape

    if hflip:
        length = play_history.__len__()
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
        logging.info(msg=f'data augment hflip: size {length} -> {play_history.__len__()}')

    if vflip:
        length = play_history.__len__()
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
        flip_action_list = list(flip_action_array)

        flip_winner_list = [winner for winner in play_history.winner_list]
        del size

        play_history.append(state_list=flip_state_list, action_list=flip_action_list, winner_list=flip_winner_list)
        logging.info(msg=f'data augment hflip: size {length} -> {play_history.__len__()}')
    
    play_history.data_check()

    return play_history