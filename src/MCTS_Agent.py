import copy
import math
import random
from Random_Agent import Random
from Agents import Agent
from game_utils import get_legal_indices, playout


class Node():
	"""A class representing a node in the Monte Carlo Tree Search (MCTS) tree."""

	def __init__(self, bitboard, env, move=None, parent=None):
		"""Initialize an MCTS node.

		Args:
			bitboard (int): Unique key representing the game state (x_bitboard << 9 | y_bitboard).
			env (Tic_Tac_Toe_State): The game environment at this node.
			move (int, optional): The move (0-8) that led to this node.
			parent (Node, optional): The parent node in the MCTS tree.
		"""
		self.env = env          # Game environment
		self.move = move        # Move that created this node
		self.key = bitboard     # Unique state identifier
		self.parent = parent    # Parent node

		self.children = []      # List of child nodes
		self.expanded = False   # Whether the node has been fully expanded
		self.visits = 0         # Number of times this node has been visited
		self.wins = 0           # Total reward accumulated from simulations

	def UCB1(self, exploration_constant=1.41):
		"""Calculate the UCB1 value for this node to balance exploration and exploitation.

        Args:
            exploration_constant (float): Constant controlling exploration (default: sqrt(2) ≈ 1.41).

        Returns:
            float: UCB1 score; returns infinity if the node has not been visited.
        """
		if self.visits == 0:
			return float('inf')
		exploitation = self.wins / self.visits  # Average reward
		exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)  # Exploration term
		return exploitation + exploration

	def selectChild(self):
		"""Select the child node with the highest UCB1 score.

       Returns:
           Node: The selected child node.
       """
		return max(self.children, key=lambda child: child.UCB1())


class MCTS(Agent):
	"""A class implementing a Monte Carlo Tree Search (MCTS) agent for Tic-Tac-Toe."""

	def __init__(self, num_simulations=50):
		"""Initialize the MCTS agent.

        Args:
            num_simulations (int): Number of simulations to run per move.
        """
		super().__init__()  # Call the parent class (Agent) initializer
		self.agentType = "MCTS"  # Set the agent type identifier
		self.num_simulations = num_simulations  # Number of MCTS simulations

		# Initialize random agents for rollout simulations
		self.rollout_agent_1 = Random()
		self.rollout_agent_2 = Random()

	def move(self, observation):
		"""Select the best move using MCTS.

        Args:
            observation (list): The current game state, containing [bitboards, turn, move_number, env].

        Returns:
            int: The index (0-8) of the chosen move.
        """

		# Create a copy of the environment and disable rendering
		environment = copy.deepcopy(observation[3])
		render_mode = environment.render_mode
		environment.render_mode = False

		# Extract bitboards and current player
		x_bitboard, y_bitboard, _ = observation[0]
		root_piece = observation[1]                 # Current player's turn (0 for x, 1 for o)
		root_key = x_bitboard << 9 | y_bitboard     # Unique state key
		root_node = Node(root_key, environment)     # Root node for MCTS

		# Run MCTS simulations
		env = copy.deepcopy(environment)
		for _ in range(self.num_simulations):
			node = self.select(env, root_node)              # Select a node to simulate
			rewards = self.rollout(node)                    # Simulate a game from the node
			self.backpropagate(node, rewards, root_piece)   # Update node statistics

		# Choose the child with the most visits
		best_child = max(root_node.children, key=lambda child: child.visits)

		# Restore original rendering mode
		environment.render_mode = render_mode

		return best_child.move

	def select(self, env, root):
		"""Select a node for expansion or simulation using the UCB1 policy.

        Args:
            env (Tic_Tac_Toe_State): The current game environment.
            root (Node): The root node of the MCTS tree.

        Returns:
            Node: The selected node for simulation.
        """
		node = root
		state = env
		piece = 1  # Current player for simulation
		p = 0  # Player index (toggles between 0 and 1)
		while not state.done:
			p = 1 - p  # Switch player
			if not node.expanded:
				# Expand the node by adding all legal moves as children
				legalMoves = get_legal_indices(state.bitboards)
				for move in legalMoves:
					new_state = copy.deepcopy(state)
					new_state.step(move)  # Apply move to new state
					x_bitboard, y_bitboard, _ = new_state.bitboards
					bit_board = x_bitboard << 9 | y_bitboard  # Create state key
					new_node = Node(bit_board, new_state, move, node)
					node.children.append(new_node)
				node.expanded = True
				child = random.choice(node.children)  # Randomly select a new child
				return child if p == piece else node

			# Select the best child based on UCB1
			node = node.selectChild()
			state = node.env

		return node if p == piece else node.parent

	def rollout(self, start):
		"""Simulate a random game from the given node to completion.

        Args:
            start (Node): The starting node for the rollout.

        Returns:
            list: Rewards for both players from the simulated game.
        """
		return playout(copy.deepcopy(start.env), [self.rollout_agent_1, self.rollout_agent_2])

	def backpropagate(self, leaf, rewards, index):
		"""Update node statistics by backpropagating rewards.

        Args:
            leaf (Node): The leaf node where the simulation started.
            rewards (list): Rewards for both players from the simulation.
            index (int): The index of the current player (0 for x, 1 for o).
        """
		node = leaf
		p = 1 - index  # Opponent's index
		while node.parent is not None:
			p = 1 - p  # Switch player
			node.visits += 1
			node.wins += rewards[p]  # Accumulate reward
			node = node.parent
		p = 1 - p
		node.visits += 1
		node.wins += rewards[p]  # Update root node
