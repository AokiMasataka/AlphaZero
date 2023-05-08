import pickle
import numpy as np


class PlayHistory:
    def __init__(self, state: list[np.ndarray], actions: list[int], winner: list[int]):
        self.state = state
        self.actions = actions
        self.winner = winner

    def __len__(self):
        return self.state.__len__()
    
    def __call__(self):
        pass

    def valid(self) -> None:
        sl, al, wl = len(self.state), len(self.actions), len(self.winner)
        assert sl == al == wl, f'state: {sl} action: {al} winner: {wl}'
    
    def append(self, other) -> None:
        self.state += other.state
        self.actions += other.actions
        self.winner += other.winner

    def split_data(self, train_rate: float = 0.8):
        split_point = int(self.__len__() * train_rate)

        train_history = PlayHistory(
            state=self.state[:split_point],
            actions=self.actions[:split_point],
            winner=self.winner[:split_point]
        )
        valid_history = PlayHistory(
            state=self.state[split_point:],
            actions=self.actions[split_point:],
            winner=self.winner[split_point:],
        )
        return train_history, valid_history

    
    def save_history(self, history_path: str) -> None:
        with open(history_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_history(history_path: str):
        with open(history_path, 'rb') as f:
            history = pickle.load(f)
    
        return history
