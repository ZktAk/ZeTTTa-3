import pickle
from Agents import Agent


class Optimal(Agent):
    """A class representing an optimal Tic-Tac-Toe agent that uses a precomputed move dictionary."""

    def __init__(self, path="optimal_moves_dictionary.pkl"):
        """Initialize the Optimal agent by loading a precomputed move dictionary.

        Args:
            path (str): Path to the pickled dictionary file containing optimal moves.
        """
        super().__init__()  # Call the parent class (Agent) initializer
        self.agentType = "Optimal"  # Set the agent type identifier
        with open(path, 'rb') as f:
            loaded_dict = pickle.load(f)  # Load the precomputed move dictionary
        self.dictionary = loaded_dict  # Store the dictionary for move lookups

    def move(self, observation):
        """Select the optimal move based on the current game state.

        Args:
            observation (list): The current game state, where observation[0] contains the bitboards [x, o, empty].

        Returns:
            int: The index (0-8) of the optimal move.
        """
        # Extract bitboards from observation
        x_bitboard, y_bitboard, _ = observation[0]

        # Create a unique state key by combining x and o bitboards
        key = x_bitboard << 9 | y_bitboard

        # Look up the optimal move in the dictionary
        try:
            move = self.dictionary[key]
        except KeyError:
            # Print bitboards for debugging if the key is not found
            from game_utils import pretty_bin
            print(f"{pretty_bin(x_bitboard, 9)} \n{pretty_bin(y_bitboard, 9)} \n{pretty_bin(_, 9)}")

        # Convert the move (bitmask) to a board index (0-8)
        n = 0
        while move > 1:
            move = move >> 1
            n += 1

        return n