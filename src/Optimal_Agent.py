import pickle
from Agents import Agent


class Optimal(Agent):
    def __init__(self, path="optimal_moves_dictionary.pkl"):
        super().__init__()
        self.agentType = "Optimal"
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)
        self.dictionary = loaded_dict

    def move(self, observation):
        x_bitboard, y_bitboard, _ = observation[0]

        key = x_bitboard << 9 | y_bitboard
        try:
            move = self.dictionary[key]
        except KeyError:
            from BinHelp import pretty_bin
            print(f"{pretty_bin(x_bitboard, 9)} \n{pretty_bin(y_bitboard, 9)} \n{pretty_bin(_, 9)}")

        n = 0
        while move > 1:
            move = move >> 1
            n += 1

        return n