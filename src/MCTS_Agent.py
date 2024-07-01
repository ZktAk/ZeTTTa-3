import copy
import math
import ctypes
import Agents
from main import playout


class Node():
	def __init__(self, bitboard, env, piece, move=None, parent=None):
		self.env = env
		self.move = move
		self.piece = piece
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


class MCTS():

	def __init__(self, num_simulations=50):
		self.agentType = "MCTS"
		self.num_simulations = num_simulations

	def give_reward(self, reward):
		pass


	def move(self, state, piece):
		root_key = state.x_bitboard << 9 | state.y_bitboard
		root_node = Node(root_key, state, piece)


		env = copy.deepcopy(state)

		for n in range(self.num_simulations):
			node = self.select(env, root_node, copy.deepcopy(piece))
			rewards = self.simulate(node)
			self.backpropagate(node, rewards)
		best_child = max(root_node.children, key=lambda child: child.visits)
		return best_child.move


	def select(self, env, root, piece):
		node = root
		state = env
		p = 1 - piece
		while not state.isTerminal():
			p = 1 - p
			if not node.expanded:
				legalMoves = state.getLegalMoves()
				for move in legalMoves:
					new_state = copy.deepcopy(state)
					new_state.move(move, p)
					bit_board = new_state.x_bitboard << 9 | new_state.y_bitboard
					new_node = Node(bit_board, new_state, p, move, node)
					node.children.append(new_node)
				node.expanded = True
				if node.children[0].piece == piece:
					return node.children[0]
				else:
					return node

			node = node.selectChild()
			state = node.env

		if node.piece == piece:
			return node
		else:
			return node.parent

	def simulate(self, start):
		p1 = Agents.Agent("Randy", 0)
		p2 = Agents.Agent("Randy", 1)
		rewards = playout(copy.deepcopy(start.env), p1, p2, 0b0)
		return rewards

	def backpropagate(self, leaf, rewards):
		node = leaf
		while node.parent is not None:
			node.visits += 1
			node.wins += rewards[leaf.piece]
			node = node.parent
		node.visits += 1
		node.wins += rewards[leaf.piece]

