import random
from BinHelp import get_legal_indices
from Agents import Agent


class Random(Agent):

	def __init__(self):
		super().__init__()
		self.agentType = "Random"

	def move(self, observation, env=None):
		legal_squares = get_legal_indices(observation[0])
		return random.choice(legal_squares)


