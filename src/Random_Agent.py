import random


class Random:

	def __init__(self, extra_params):
		self.agentType = "Random"

	def move(self, state, piece):
		return random.choice(state.getLegalMoves())

	def give_reward(self, reward):
		pass
