import random
from game_utils import get_legal_indices
from Agents import Agent


class Random(Agent):
	"""A class representing a random Tic-Tac-Toe agent that selects moves randomly from legal positions."""

	def __init__(self):
		"""Initialize the Random agent."""
		super().__init__()  # Call the parent class (Agent) initializer
		self.agentType = "Random"  # Set the agent type identifier

	def move(self, observation):
		"""Select a random legal move based on the current game state.

        Args:
            observation (list): The current game state, where observation[0] contains the bitboards.

        Returns:
            int: The index (0-8) of the randomly chosen legal square.
        """
		legal_squares = get_legal_indices(observation[0])  # Get list of legal move indices
		return random.choice(legal_squares)  # Return a randomly selected legal move


