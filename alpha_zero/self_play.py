from copy import deepcopy
import math
import numpy as np


def softmax_with_temp(x, temp=1.0):
    x = np.exp(x * temp)
    return x / np.sum(x)


class Node:
    def __init__(self, state, p):
        self.state = state
        self.p = p
        self.n = 1
        self.w = 0.0

        self.is_leaf = True
        self.children = []

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


class MCTC:
    def __init__(self, model, game, root_state, c_puct=1.0, temp=1.0):
        self.model = model.eval()
        self.c_puct = c_puct
        self.temp = temp

        self.game = game(root_state)
        self.root_node = Node(root_state, p=1.0)
        self.nodes = {self.game.hash(): self.root_node}

    def solver(self, num_searchs):
        for _ in range(num_searchs):
            pass

    def _move_to_leaf(self):
        state_hash_trace = []
        current_node = self.root_node

        while not current_node.is_leaf:
            state_hash_trace.append(current_node.hash())
            pn = math.sqrt(current_node.n)
            scores = np.array([self.nodes[child].score(pn, self.c_puct) for child in current_node.children], dtype=np.float32)

            scores = softmax_with_temp(scores, self.temp)
            action = np.random.choice(current_node.__len__(), p=scores)
            current_node = self.nodes[current_node.children[action]]

        return state_hash_trace, current_node

    def _extend(self, current_node):
        self.game.__init__(state=current_node())

        legal_action = self.game.get_legal_action()

        value, policy = self.model(self.game.encode_state())
        value, policy = value.squeeze(0).numpy(), policy.squeeze(0).numpy()

        for action, p in zip(legal_action, policy[legal_action]):
            state = self.game.get_next_state(action=action)
            _hash = state.tobytes()
            self.nodes[_hash] = Node(state=state, p=p)
            current_node.children.append(_hash)
        return value
