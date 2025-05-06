import random
from Agents import Agent


class QTable(Agent):
	"""A class representing a Q-learning Tic-Tac-Toe agent that uses a Q-table to select moves."""

	def __init__(self, convergence, num_games):
		"""Initialize the QTable agent with convergence and number of games.

        Args:
            convergence (float): The convergence factor for exploration decay.
            num_games (int): The total number of games for exploration decay calculation.

        The exploration gamma is calculated as convergence^(1/num_games).
        """
		super().__init__()  # Call the parent class (Agent) initializer
		self.agentType = "QTable"  # Set the agent type identifier

		self.p = 1  # Initial exploration probability (starts at 1, decays over time)
		self.gamma = convergence ** (1 / float(num_games))  # Exploration decay factor
		self.move_dictionary = {}  # Dictionary mapping state keys to move scores and actions
		self.game_memory = []  # List to store state-action pairs for the current game


	def move(self, observation):
		"""Select a move based on the current game state using Q-learning.

        Args:
            observation (list): The current game state, where observation[0] contains the bitboards [x, o, empty].

        Returns:
            int: The index (0-8) of the chosen move.
        """
		# Extract bitboards from observation
		x_bitboard, y_bitboard, empty_bitboard = observation[0]

		# Create a unique state key by combining x and o bitboards
		state_key = x_bitboard << 9 | y_bitboard

		# Check if the state exists in the move dictionary; if not, initialize it
		try:
			moves = self.move_dictionary[state_key]
		except KeyError:
			moves = {}
			# Initialize moves with default scores for legal squares
			for n in range(9):
				if (0b1 << n) & empty_bitboard > 0:
					moves[0.1*n] = n  # Assign initial score based on position
			self.move_dictionary[state_key] = moves

		# Decide whether to explore (random move) or exploit (best move)
		if random.random() <= self.p:
			# Exploration: choose a random move
			action_key = random.choice(list(self.move_dictionary[state_key].keys()))
		else:
			# Exploitation: choose the move with the highest score
			move_scores = list(self.move_dictionary[state_key].keys())
			move_scores.sort(reverse=True)
			action_key = move_scores[0]

		# Store the state and action in game memory
		self.game_memory.append([state_key, action_key])
		# Retrieve the action (board position) corresponding to the chosen action key
		action = self.move_dictionary[state_key][action_key]
		return action

	def give_reward(self, reward):
		"""Update the Q-table based on the received reward.

        Args:
            reward (float): The reward for the game (0 for loss, 0.5 for draw, 1 for win).
        """
		# Convert reward to Q-table update score: 0 -> -2 (loss), 0.5 -> 0 (draw), 1 -> 1 (win)
		score = -2 if reward==0 else (reward*2)-1

		# Update the Q-table for each state-action pair in the game memory
		for state in self.game_memory:
			state_key = state[0]
			new_action_value = state[1] + score  # Update the action value
			# Update the move dictionary with the new score, preserving the action
			self.move_dictionary[state_key][new_action_value] = self.move_dictionary[state_key].pop(state[1])

		# Clear game memory for the next game
		self.game_memory = []
		# Decay exploration probability
		self.p *= self.gamma