import pickle
import logging
import numpy as np


class PlayHistory:
    def __init__(self, state_list: list = None, action_list: list[int] = None, winner_list: list[int] = None):
        if state_list is not None:
            self.state_list = state_list
            self.action_list = action_list
            self.winner_list = winner_list
            self._data_check()
        else:
            self.state_list = list()
            self.action_list = list()
            self.winner_list = list()

    def __len__(self):
        return self.state_list.__len__()
    
    def __add__(self, other):
        other._data_check()
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
    
    def _data_check(self):
        assert len(self.state_list) == len(self.action_list) == len(self.winner_list), \
            f'histry length error: ' \
            f'{self.state_list.__len__()} - ' \
            f'{self.action_list.__len__()} - ' \
            f'{self.winner_list.__len__()}'
        logging.debug(msg='PlayerHitory: checked validation')
    
    def append(self, state_list=None, action_list=None, winner_list=None, history=None):
        if history is not None:
            assert isinstance(history, PlayHistory), 'history class'
            history._data_check()
            self.state_list += history.state_list
            self.action_list += history.action_list
            self.winner_list += history.winner_list
        else:
            assert state_list.__len__() == action_list.__len__() == winner_list.__len__(), \
                f'histry length error: ' \
                f'{state_list.__len__()} - ' \
                f'{action_list.__len__()} - ' \
                f'{winner_list.__len__()}'
            self.state_list += state_list
            self.action_list += action_list
            self.winner_list += winner_list
    
    def save_history(self, save_path):
        histry_dict = {
            'state_list': self.state_list,
            'action_list': self.action_list,
            'winner_list': self.winner_list
        }

        with open(save_path, 'wb') as f:
            pickle.dump(histry_dict, f)

    def load_history(self, laod_path):
        with open(laod_path, 'rb') as f:
            histry_dict = pickle.load(f)

        self.state_list = histry_dict['state_list']
        self.action_list = histry_dict['action_list']
        self.winner_list = histry_dict['winner_list']
    
    def get_train_valid_data(self, train_data_rate=0.8):
        rand_indexes = np.random.choice(self.__len__(), self.__len__(), replace=False)
        train_indexes = rand_indexes[0: int(self.__len__() * train_data_rate)]
        valid_indexes = rand_indexes[int(self.__len__() * train_data_rate): self.__len__()]
        
        state_array = np.stack(self.state_list, axis=0)
        action_array = np.stack(self.action_list, axis=0)
        winner_array = np.stack(self.winner_list, axis=0)

        train_history = PlayHistory(
            state_list=list(state_array[train_indexes]),
            action_list=list(action_array[train_indexes]),
            winner_list=list(winner_array[train_indexes])
        )

        if train_data_rate == 1:
            valid_history = None
        else:
            valid_history = PlayHistory(
                state_list=list(state_array[valid_indexes]),
                action_list=list(action_array[valid_indexes]),
                winner_list=list(winner_array[valid_indexes])
            )
        logging.info(msg=f'train data size: {train_history.__len__()} valid data size: {valid_history.__len__()}')
        return train_history, valid_history



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