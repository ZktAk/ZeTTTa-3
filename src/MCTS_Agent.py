import copy
import math
import random
from Random_Agent import Random
from Agents import Agent
from main import playout
from BinHelp import get_legal_indices


class Node():
	def __init__(self, bitboard, env, move=None, parent=None):
		self.env = env
		self.move = move
		self.key = bitboard
		self.parent = parent

		self.children = []
		self.expanded = False
		self.visits = 0
		self.wins = 0

	def UCB1(self, exploration_constant=1.41):
		if self.visits == 0:
			return float('inf')
		exploitation = self.wins / self.visits
		exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
		return exploitation + exploration

	def selectChild(self):
		selected_child = max(self.children, key=lambda child: child.UCB1())
		return selected_child


class MCTS(Agent):

	def __init__(self, num_simulations=50):
		super().__init__()
		self.agentType = "MCTS"
		self.num_simulations = num_simulations

		self.rollout_agent_1 = Random()
		self.rollout_agent_2 = Random()


	def move(self, observation, environment=None):

		render_mode = environment.render_mode
		environment.render_mode = False

		x_bitboard, y_bitboard, _ = observation[0]
		root_piece = observation[1]

		root_key = x_bitboard << 9 | y_bitboard
		root_node = Node(root_key, environment)

		env = copy.deepcopy(environment)

		for n in range(self.num_simulations):
			node = self.select(env, root_node)
			rewards = self.rollout(node)
			self.backpropagate(node, rewards, root_piece)
		best_child = max(root_node.children, key=lambda child: child.visits)

		environment.render_mode = render_mode

		return best_child.move


	def select(self, env, root):
		node = root
		state = env
		piece = 1
		p = 0
		while not state.done:
			p = 1 - p
			if not node.expanded:
				legalMoves = get_legal_indices(state.bitboards)
				for move in legalMoves:
					new_state = copy.deepcopy(state)
					new_state.step(move)

					x_bitboard, y_bitboard, _ = new_state.bitboards
					bit_board = x_bitboard << 9 | y_bitboard

					new_node = Node(bit_board, new_state, move, node)
					node.children.append(new_node)
				node.expanded = True
				child = random.choice(node.children)
				if p == piece:
					return child
				else:
					return node

			node = node.selectChild()
			state = node.env

		if p == piece:
			return node
		else:
			return node.parent

	def rollout(self, start):
		rewards = playout(copy.deepcopy(start.env), [self.rollout_agent_1, self.rollout_agent_2])
		return rewards

	def backpropagate(self, leaf, rewards, index):
		node = leaf
		p = 1 - index
		while node.parent is not None:
			p = 1 - p
			node.visits += 1
			node.wins += rewards[p]
			node = node.parent
		p = 1 - p
		node.visits += 1
		node.wins += rewards[p]
