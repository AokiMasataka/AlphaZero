import os
import ray
import pickle
import logging
from copy import deepcopy
from tqdm import tqdm
import math
import numpy as np


def softmax(x, temperature=1.0):
    f = np.exp((x - np.max(x) / temperature))
    return f / f.sum(axis=0)


class Node:
    def __init__(self, state, p, parent=None):
        self.state = state
        self.p = p
        self.w = 0.0
        self.n = 0

        self.is_leaf = True
        self.children = []
        self.parent = parent

    def __len__(self):
        return self.children.__len__()

    def __call__(self):
        return deepcopy(self.state)

    def score(self, pn, c_puct):
        u = c_puct * self.p * math.sqrt(pn) / (1 + self.n)
        if self.n == 0:
            q = self.w
        else:
            q = self.w / self.n
        return q + u


def search(root_state, model, game_module, num_searchs, c_puct=1.0, temperature=1.0):
    root_node = Node(state=root_state, p=1.0, parent=None)

    for _ in range(num_searchs):
        value = _move_to_leaf(node=root_node, game_module=game_module, model=model, c_puct=c_puct, temperature=temperature)
        root_node.w += value
        root_node.n += 1
    
    policy = np.array([child.n / root_node.n for child in root_node.children], dtype=float)
    policy = softmax(policy)
    return policy


def _move_to_leaf(node, model, game_module, c_puct, temperature):
    if node.is_leaf:
        node.is_leaf = False
        legal_actions = game_module.get_legal_action_functional(state=node.state)
        value, policy = model.inference_state(game_module.encode_state(node.state))
        if legal_actions:
            for action in legal_actions:
                state = game_module.get_next_state(state=node.state, action=action)
                node.children.append(Node(state=state, p=policy[action], parent=node))
        else:
            node.children = [Node(state=deepcopy(-node.state), p=1.0, parent=node)]
        return value
            
    elif game_module.is_done_functional(state=node.state):
        value = game_module.is_win(state=node.state)
        return value

    else:
        scores = np.array([child.score(pn=node.n, c_puct=c_puct) for child in node.children], dtype=np.float32)
        action = np.random.choice(scores.__len__(), p=softmax(scores, temperature=temperature))
        node = node.children[action]
        value = -_move_to_leaf(node=node, game_module=game_module, model=model, c_puct=c_puct, temperature=temperature)

        node.w += value
        node.n += 1
        return value


@ray.remote(num_cpus=1, num_gpus=0)
def self_play(game_module, init_dict, f_model, b_model, num_searchs, random_play=0, c_puct=1.0, temperature=1.0):
    play_history = PlayHistory()
    game = game_module(**init_dict)
    models = [f_model, b_model]

    step = 0
    while not game.is_done():
        play_history.state_list.append(game_module.encode_state(state=deepcopy(game.state)))
        legale_actions = game.get_legal_action()
        if step < random_play:
            if legale_actions:
                action = np.random.choice(a=legale_actions)
                game.action(action=action)
            else:
                action = game.pass_action()
        else:
            if legale_actions:
                policy = search(
                    root_state=game.state,
                    model=models[0],
                    game_module=game_module,
                    num_searchs=num_searchs,
                    c_puct=c_puct,
                    temperature=temperature
                )
                action = np.random.choice(a=legale_actions, p=policy)
                game.action(action=action)
            else:
                action = game.pass_action()
        
        play_history.action_list.append(action)
        models = models[::-1]

        step += 1
    
    winner = game.get_winner()
    if winner == 1:
        play_history.winner_list = list(1 if i % 2 else -1 for i in range(play_history.__len__()))
    elif winner == -1:
        play_history.winner_list = list(-1 if i % 2 else 1 for i in range(play_history.__len__()))
    else:
        play_history.winner_list = list(0 for _ in range(play_history.__len__()))
    
    return play_history


def parallel_self_play(
        f_model, b_model, num_games, game, init_dict, num_searchs, random_play=0, c_puct=1.0, temperature=1.0, num_cpus=None
):
    self_play_histry = PlayHistory()

    if num_cpus is None:
        num_cpus = os.cpu_count()
    
    ray.init(num_cpus=min(num_cpus, os.cpu_count()))
    f_model_id = ray.put(f_model)
    b_model_id = ray.put(b_model)

    work_ids = []
    for _ in range(num_games):
        work_id = self_play.remote(
            game, init_dict, f_model_id, b_model_id, num_searchs, random_play, c_puct, temperature
        )
        work_ids.append(work_id)

    winner = []
    for work_id in tqdm(work_ids):
        histry = ray.get(work_id)
        self_play_histry = self_play_histry + histry
        winner.append(histry.winner_list[0])

    ray.shutdown()
    return self_play_histry, winner


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
