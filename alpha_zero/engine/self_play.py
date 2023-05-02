import os
import ray
import copy
import math
from tqdm import tqdm
import numpy as np
from ..games.game_base import BaseGame
from ..model.scale_model import ScaleModel
from ..utility import PlayHistory


def softmax(x, temperature=1.0):
    f = np.exp((x - np.max(x) / temperature))
    return f / f.sum(axis=0)


class Node:
    def __init__(self, obj: BaseGame, p: float, parent):
        self._obj = obj
        self._is_done = self._obj.is_done()
        self.p = p
        self.w = 0.0
        self.n = 0

        self.is_leaf = True
        self.children: list[Node] = []
        self.parent: Node = parent
    
    @property
    def obj(self):
        return self._obj

    @property
    def state(self):
        return self._obj.state
    
    @property
    def is_done(self):
        return self._is_done
    
    def obj_copy(self):
        return self._obj.copy()
    
    def state_copy(self):
        return self._obj.state.copy()

    def __len__(self):
        return self.children.__len__()
    
    def __repr__(self):
        return f'n: {self.n} - w: {self.w}'
    
    def score(self, c_puct: float):
        u = c_puct * self.p * math.sqrt(self.parent.n) / (1 + self.n)
        q = (self.obj.player * self.w + 1e-6) / (self.n + 1e-6)
        return q + u


class MonteCarlo:
    def __init__(
        self,
        model: ScaleModel,
        num_searchs: int,
        c_puct: float = 1.0,
        temperature: float = 1.0

    ):
        self.model = model
        self._num_searchs = num_searchs
        self._c_puct = c_puct
        self._temperature = temperature
    
    @property
    def num_searchs(self):
        return self._num_searchs

    @property
    def c_puct(self):
        return self._c_puct
    
    @property
    def temperature(self):
        return self._temperature
    
    def set_num_searchs(self, num_searchs: int):
        self._num_searchs = num_searchs
    
    def set_c_puct(self, c_puct: float):
        self._c_puct = c_puct
    
    def set_temperature(self, temperature: float):
        self._temperature = temperature
    
    def __call__(self, root_obj: BaseGame):
        root_node = Node(obj=root_obj.copy(), p=1.0, parent=None)

        for _ in range(self._num_searchs):
            traced_nodes = []
            value, traced_nodes = self._move_to_leaf(node=root_node, traces_nodes=traced_nodes)
            self._back_to_root(value=value, traced_nodes=traced_nodes)
        
        policy = np.array([child.n / root_node.n for child in root_node.children], dtype=float)
        policy = softmax(policy)
        return policy
    
    def _move_to_leaf(self, node: Node, traces_nodes: list[Node]):
        traces_nodes.append(node)

        if node.is_done:
            value = node.obj.get_winner()
            return value, traces_nodes

        if node.is_leaf:
            node.is_leaf = False
            legal_actions = node.obj.get_legal_action()

            value, policy = self.model.inference_state(state=node.obj.encode_state())

            if 0 == legal_actions:
                new_game_obj = node.obj.change_player()
                node.children.append(Node(obj=new_game_obj, p=1.0, parent=node))
            elif 1 == legal_actions:
                new_game_obj = node.obj.action(action=legal_actions[0])
                node.children.append(Node(obj=new_game_obj, p=1.0, parent=node))
            else:    
                policy = policy[legal_actions]
                for action, p in zip(legal_actions, softmax(x=policy)):
                    new_game_obj = node.obj.action(action=action)
                    node.children.append(Node(obj=new_game_obj, p=p, parent=node))
                
            return value, traces_nodes
        else:
            if node.__len__() == 1:
                value, traces_nodes = self._move_to_leaf(node=node.children[0], traces_nodes=traces_nodes)
            else:
                scores = np.array([child.score(c_puct=self.c_puct) for child in node.children], dtype=np.float32)
                action = np.random.choice(scores.__len__(), p=softmax(x=scores, temperature=self.temperature))
                value, traces_nodes = self._move_to_leaf(node=node.children[action], traces_nodes=traces_nodes)
            return value, traces_nodes

    
    def _back_to_root(self, value: float, traced_nodes: list[Node]):
        for node in traced_nodes:
            node.w += value
            node.n += 1


@ray.remote(num_cpus=1, num_gpus=0)
def self_play(
    f_model: ScaleModel,
    b_model: ScaleModel,
    game: BaseGame,
    num_searchs: int,
    random_play: int = 0,
    c_puct: float = 1.0,
    temperature: float = 1.0,
) -> PlayHistory:
    play_history = PlayHistory()

    monte_carlo = [
        MonteCarlo(model=f_model, num_searchs=num_searchs, c_puct=c_puct, temperature=temperature),
        MonteCarlo(model=b_model, num_searchs=num_searchs, c_puct=c_puct, temperature=temperature)
    ]

    step = 0
    while not game.is_done():
        play_history.state_list.append(game.encode_state())
        legale_actions = game.get_legal_action()

        if step < random_play:
            if legale_actions:
                action = np.random.choice(a=legale_actions)
                game = game.action(action=action)
            else:
                action = game.pass_action
                game = game.change_player()
        else:
            if legale_actions:
                policy = monte_carlo[0](root_obj=game)
                action = np.random.choice(a=legale_actions, p=policy)
                game = game.action(action=action)
            else:
                action = game.pass_action
                game = game.change_player()
        
        play_history.action_list.append(action)
        monte_carlo = monte_carlo[::-1]

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
        f_model: ScaleModel,
        b_model: ScaleModel,
        game_module: BaseGame,
        init_dict: dict,
        num_games: int,
        num_searchs: int,
        random_play: int = 0,
        c_puct: float = 1.0,
        temperature: float = 1.0,
        num_cpus: int = None
):
    self_play_histry = PlayHistory()

    if num_cpus is None:
        num_cpus = os.cpu_count()
    
    num_cpus = 1
    
    ray.init(num_cpus=num_cpus)
    f_model_id = ray.put(f_model)
    b_model_id = ray.put(b_model)

    work_ids = []
    game = game_module(**init_dict)
    for _ in range(num_games):
        work_id = self_play.remote(
            f_model_id, b_model_id, game, num_searchs, random_play, c_puct, temperature
        )
        work_ids.append(work_id)

    winner = []
    for work_id in tqdm(work_ids):
        histry = ray.get(work_id)
        self_play_histry = self_play_histry + histry
        winner.append(histry.winner_list[0])

    ray.shutdown()
    return self_play_histry, winner