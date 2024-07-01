import pickle


class Optimal():
    def __init__(self, extra_params, path="optimal_moves_dictionary.pkl"):
        self.agentType = "Optimal"
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.dictionary = loaded_dict

    def move(self, state, piece):
        key = state.x_bitboard << 9 | state.y_bitboard
        move = self.dictionary[key]
        return move

    def give_reward(self, reward):
        pass
