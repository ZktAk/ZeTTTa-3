from Random_Agent import Random
from Optimal_Agent import Optimal
from QTable_Agent import QTable
import MCTS_Agent


class Agent:

	def __init__(self, model, piece, extra_params=None):
		self.piece = piece

		agents = {"Randy": Random, "Optimus": Optimal, "Quill": QTable, "HeavyTree": MCTS_Agent.MCTS}  # HeavyTree, Neura, ScoutNet
		"""
		Hidden meaning behind name choices (yes, this was completely unnecessary and took way too long):

		Randy: A playful and straightforward name, reflecting the random nature of the model's actions.
		Optimus: Stands for Optimus Prime.
		Quill: Symbolizes the model's meticulous recording and updating of its Q-table, akin to writing with a quill pen.
		HeavyTree: Yes, a TF2 reference. The Heavy symbolises the brute force nature of a Monte-Carlo Tree Search (MCTS) model.
		Neura: Stands for Neural Network (NN). Symbolizes intelligence, sophistication, and adaptability.
		ScoutNet: Another TF2 reference. If a MCTS is the Heavy and a NN is the Spy, then combined they gives us the Scout. 
		"""

		self.model = agents[model](extra_params)

		self.agentType = self.model.agentType

	def move(self, state):
		return self.model.move(state, self.piece)

	def give_reward(self, reward):
		self.model.give_reward(reward)

