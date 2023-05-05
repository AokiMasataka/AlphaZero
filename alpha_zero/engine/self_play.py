import os
import ray
from tqdm import tqdm
import numpy as np
from .mcts import MonteCarloTreeSearch
from ..games.game_base import BaseGame
from ..model.scale_model import ScaleModel
from ..utility import PlayHistory


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
        MonteCarloTreeSearch(model=f_model, num_searchs=num_searchs, c_puct=c_puct, temperature=temperature),
        MonteCarloTreeSearch(model=b_model, num_searchs=num_searchs, c_puct=c_puct, temperature=temperature)
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
                policy = monte_carlo[0](obj=game)
                action = np.random.choice(a=legale_actions, p=policy)
                game = game.action(action=action)
            else:
                action = game.pass_action
                game = game.change_player()
        
        play_history.action_list.append(action)
        monte_carlo = monte_carlo[::-1]

        step += 1

    winner = game.get_winner()
    winner = winner * game.player
    play_history.winner_list = list(winner if i % 2 else -winner for i in range(play_history.__len__()))
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