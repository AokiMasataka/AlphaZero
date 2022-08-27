import ray
from copy import deepcopy
import math
import numpy as np

from data import PlayHistory


def softmax_with_temp(x, temperature=1.0):
    exp = np.exp(x / temperature)
    f_x = exp / np.sum(exp)
    return f_x


class Node:
    def __init__(self, state, p, parent=None, player=1):
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

    def score(self, pn, c_puct, player):
        u = c_puct * self.p * math.sqrt(pn) / self.n
        q = (player * self.w) / self.n
        return q + u


def mcts_search(root_state, model, game, num_searchs, c_puct=1.0, temperature=1.0):
    game = game()

    def _move_to_leaf(node, player):
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
                    state = game.next_state(state=deepcopy(node.state), action=action)
                    node.children.append(Node(state=deepcopy(state), p=p, parent=node))
                node.is_leaf = False
            return node, value
        else:
            pn = node.n
            scores = np.array([child.score(pn=pn, c_puct=c_puct, player=player) for child in node.children], dtype=np.float32)
            scores = softmax_with_temp(x=scores, temperature=temperature)
            action = np.random.choice(node.__len__(), p=scores)
            node = node.children[action]
            return _move_to_leaf(node=node, player=player * -1)

    def _back_to_root(node, value):
        node.w += value
        node.n += 1
        if node.parent:
            _back_to_root(node=node.parent, value=value)
        else:
            return 0

    root_node = Node(state=deepcopy(root_state), p=1.0, player=1)

    for _ in range(num_searchs):
        leaf_node, value = _move_to_leaf(node=root_node, player=1)
        state_code = _back_to_root(node=leaf_node, value=value)

    child_n = np.array([child.n for child in root_node.children], dtype=np.float32)
    child_scores = softmax_with_temp(child_n, temperature=1.0)
    return np.random.choice(child_scores.__len__(), p=child_scores)


@ray.remote
def self_play(game, init_dict, model, num_searchs, random_play=0, c_puct=1.0, temperature=1.0):
    play_history = PlayHistory()
    _game = game
    game = game(**init_dict)

    for _ in range(random_play):
        if game.is_done():
            return play_history

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
                root_state=game.state, model=model, game=_game, num_searchs=num_searchs, c_puct=c_puct, temperature=temperature
            )
            action = legal_action[action_index]
            game.action(action=action)
        else:
            game.play_chenge()
            action = game.pass_action()

        play_history.action_list.append(action)

    play_history.winner_list = [i % 2 for i in range(1, play_history.__len__() + 1)]
    return play_history


def parallel_self_play(model, num_searchs, num_games, game, init_dict, random_play=0, c_puct=1.0, temperature=1.0):
    self_play_histry = PlayHistory()
    model_id = ray.put(model)

    # ray.init(ignore_reinit_error=True)
    work_in_progresses = [self_play.remote(
        game=game,
        init_dict=init_dict,
        model=model_id,
        num_searchs=num_searchs,
        random_play=random_play,
        c_puct=c_puct,
        temperature=temperature
    ) for _ in range(num_games)]

    for i in range(num_games):
        finished, work_in_progresses = ray.wait(work_in_progresses, num_returns=1)
        orf = finished[0]

        histry = ray.get(orf)
        self_play_histry = self_play_histry + histry

    return self_play_histry
