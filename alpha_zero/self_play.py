import os

import ray
from copy import deepcopy
from tqdm import tqdm
import math
import numpy as np

from data import PlayHistory


def softmax_with_temp(x, temperature=1.0):
    f = np.exp((x - np.max(x)) / temperature)
    return f / f.sum(axis=0)


class Node:
    def __init__(self, state, p, parent=None):
        self.state = state
        self.p = p
        self.w = 0.0
        self.n = 1

        self.is_leaf = True
        self.children = []
        self.parent = parent

    def __len__(self):
        return self.children.__len__()

    def __call__(self):
        return deepcopy(self.state)

    def hash(self):
        return self.state.tobytes()

    def score(self, pn, c_puct):
        u = c_puct * self.p * math.sqrt(pn) / self.n
        q = self.w / self.n
        return q + u


def mcts_search(
        root_state,
        model,
        game_module,
        init_dict,
        num_searchs,
        c_puct=1.0,
        temperature=1.0,
        return_value=False,
        mode='self_play'
):
    assert mode in ('play', 'self_play')
    game = game_module(**init_dict)

    def _move_to_leaf(node):
        if node.is_leaf:
            legal_action = game.get_legal_action(state=node.state)

            if not legal_action:
                value = model.get_value(state=node.state)
                state = deepcopy(node.state[::-1])
                node.children = [Node(state=deepcopy(state), p=1.0, parent=node)]
                node.is_leaf = False
            else:
                value, policy = model.inference_state(state=node.state)
                for action, p in zip(legal_action, policy[legal_action]):
                    state = game_module.get_next_state(state=deepcopy(node.state), action=action)
                    node.children.append(Node(state=deepcopy(state), p=p, parent=node))
                node.is_leaf = False
            return node, value
        else:
            scores = np.array([child.score(pn=node.n, c_puct=c_puct) for child in node.children], dtype=np.float32)
            action = np.argmax(scores)
            node = node.children[action]
            return _move_to_leaf(node=node)

    def _back_to_root(node, value):
        node.w += value
        node.n += 1
        if node.parent:
            _back_to_root(node=node.parent, value=-value)
        else:
            return 0

    root_node = Node(state=deepcopy(root_state), p=1.0)

    for _ in range(num_searchs):
        leaf_node, value = _move_to_leaf(node=root_node)
        game.__init__(state=leaf_node.state)
        if game.is_done():
            if np.sum(leaf_node.state[0]) < np.sum(leaf_node.state[1]):
                value = -1
            elif np.sum(leaf_node.state[1]) < np.sum(leaf_node.state[0]):
                value = 1
            else:
                value = 0
        state_code = _back_to_root(node=leaf_node, value=value)

    child_n = np.array([child.n for child in root_node.children], dtype=np.float32)
    if mode == 'self_play':
        child_scores = softmax_with_temp(child_n, temperature=temperature)
        action = np.random.choice(child_scores.__len__(), p=child_scores)
    else:
        action = np.argmax(child_n)

    if return_value:
        return action, root_node.w
    else:
        return action


@ray.remote
def self_play(game_module, init_dict, model, num_searchs, random_play=0, c_puct=1.0, temperature=1.0):
    play_history = PlayHistory()
    game = game_module(**init_dict)

    for _ in range(random_play):
        if game.is_done():
            break

        play_history.state_list.append(deepcopy(game.state))
        legale_actions = game.get_legal_action()
        if legale_actions:
            action = np.random.choice(a=legale_actions)
            game.action(action=action)
        else:
            action = game.pass_action()
        play_history.action_list.append(action)

    while not game.is_done():
        play_history.state_list.append(deepcopy(game.state))

        legal_action = game.get_legal_action(state=None)

        if legal_action:
            action_index = mcts_search(
                root_state=game.state,
                model=model,
                init_dict=init_dict,
                game_module=game_module,
                num_searchs=num_searchs,
                c_puct=c_puct,
                temperature=temperature
            )
            action = legal_action[action_index]
            game.action(action=action)
        else:
            game.play_chenge()
            action = game.pass_action()
        play_history.action_list.append(action)

    if game.winner() == 1:
        play_history.winner_list = list(1 if i % 2 else -1 for i in range(play_history.__len__()))
    elif game.winner() == -1:
        play_history.winner_list = list(-1 if i % 2 else 1 for i in range(play_history.__len__()))
    else:
        play_history.winner_list = list(0 for _ in range(play_history.__len__()))

    # if play_history.__len__() % 2 == 0:
    #     if game.is_win():
    #         play_history.winner_list = list(1 if i % 2 else -1 for i in range(play_history.__len__()))
    #     else:
    #         play_history.winner_list = list(-1 if i % 2 else 1 for i in range(play_history.__len__()))
    # else:
    #     if game.is_win():
    #         play_history.winner_list = list(-1 if i % 2 else 1 for i in range(play_history.__len__()))
    #     else:
    #         play_history.winner_list = list(1 if i % 2 else -1 for i in range(play_history.__len__()))

    return play_history


def parallel_self_play(model, num_searchs, num_games, game, init_dict, random_play=0, c_puct=1.0, temperature=1.0):
    self_play_histry = PlayHistory()

    ray.init(num_cpus=os.cpu_count() - 2)
    model_id = ray.put(model)

    work_ids = []
    for _ in tqdm(range(num_games)):
        work_id = self_play.remote(game, init_dict, model_id, num_searchs, random_play, c_puct, temperature)
        work_ids.append(work_id)

    for work_id in work_ids:
        histry = ray.get(work_id)
        self_play_histry = self_play_histry + histry

    ray.shutdown()
    return self_play_histry


def models_play(first_model, back_model, game_module, init_dict, num_searchs, random_play=0, c_puct=1.0, temperature=1.0):
    game = game_module(**init_dict)

    step = 0
    for _ in range(random_play):
        if game.is_done():
            break

        legale_actions = game.get_legal_action()
        if legale_actions:
            action = np.random.choice(a=legale_actions)
            game.action(action=action)
        else:
            game.play_chenge()

        step += 1

    while not game.is_done():
        legal_action = game.get_legal_action(state=None)

        if step % 2 == 0:
            play_model = first_model
        else:
            play_model = back_model

        if legal_action:
            action_index = mcts_search(
                root_state=game.state,
                model=play_model,
                init_dict=init_dict,
                game_module=game_module,
                num_searchs=num_searchs,
                c_puct=c_puct,
                temperature=temperature
            )
            action = legal_action[action_index]
            game.action(action=action)
        else:
            game.play_chenge()

        step += 1

    if game.winner() == 1:
        return 1
    elif game.winner() == -1:
        return 0
    else:
        return 0
