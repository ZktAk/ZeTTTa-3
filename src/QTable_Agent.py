import random


class QTable:
	def __init__(self, extra_params):

		convergence = extra_params[0]
		num_games = extra_params[1]

		"""
		Initializes a Q-table agent with convergence and number of games.

				Qtable(self, extra_params=[C, N])

				C	--	convergence

				N	--	number of games

				C^(1/N)	--	yields exploration gamma
		"""

		self.agentType = "QTable"

		self.p = 1
		self.gamma = convergence ** (1 / float(num_games))
		self.move_dictionary = {}
		self.game_memory = []


	def move(self, state, piece):

		state_key = state.x_bitboard << 9 | state.y_bitboard

		try:
			moves = self.move_dictionary[state_key]
		except KeyError:
			moves = {}
			empty_bitboard = state.getLegalMoves("bitboard")
			for n in range(9):
				if (0b1 << n) & empty_bitboard > 0:
					moves[0.1*n] = 0b1 << n
			self.move_dictionary[state_key] = moves

		if random.random() <= self.p:
			action_key = random.choice(list(self.move_dictionary[state_key].keys()))
		else:
			move_scores = list(self.move_dictionary[state_key].keys())
			move_scores.sort(reverse=True)
			action_key = move_scores[0]

		self.game_memory.append([state_key, action_key])
		action = self.move_dictionary[state_key][action_key]
		return action

	def give_reward(self, reward):

		score = -2 if reward==0 else (reward*2)-1  # 0 becomes -2, 0.5 becomes 0, and 1 stays 1
		for state in self.game_memory:
			state_key = state[0]
			new_action_value = state[1] + score
			self.move_dictionary[state_key][new_action_value] = self.move_dictionary[state_key].pop(state[1])

		self.game_memory = []
		self.p *= self.gamma