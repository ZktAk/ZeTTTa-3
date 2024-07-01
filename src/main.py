from Environment import Tic_Tac_Toe_State
import Agents
import matplotlib.pyplot as plt
import random


def playout(current_state, player1, player2, player_to_move, display=False):

	"""if current_state.draw or current_state.win:
		print("ERROR: main.playout(): \'State is already terminal\'")
		current_state.print_board()
		return"""
	reward = [0, 0]

	if current_state.draw:
		if display: print("DRAW")
		if display: current_state.print_board()
		reward = [0.5, 0.5]
		return reward
	if current_state.win:
		if display: print("WIN")
		reward[player_to_move] = 1
		if display: current_state.print_board()
		return reward


	current_player = player_to_move

	players = [player1, player2]

	if display: current_state.print_board()

	while 1:

		player = players[current_player]
		square = player.move(current_state)
		current_state.move(square, player.piece)

		if display: print("player to move: {}".format(player.piece))

		if current_state.draw:
			if display: print("DRAW")
			if display: current_state.print_board()
			reward = [0.5, 0.5]
			return reward
		if current_state.win:
			if display: print("WIN")
			reward[current_player] = 1
			if display: current_state.print_board()
			return reward

		if display: current_state.print_board()

		current_player = 1 - current_player


def play(current_state, player1, player2, player_to_move, track, display=True):

	# track: The index of the player whose stats we want to track

	players = [player1, player2]
	rewards = playout(current_state, player1, player2, player_to_move, display)

	players[0].give_reward(rewards[0])
	players[1].give_reward(rewards[1])

	current_state.reset()

	if rewards[track] == 0:
		return 0, 0, 1
	if rewards[track] == 0.5:
		return 0, 1, 0
	if rewards[track] == 1:
		return 1, 0, 0


if __name__ == '__main__':

	#random.seed(1)

	ENV = Tic_Tac_Toe_State()
	q_table_model_params = [0.1, 5000]
	p1 = Agents.Agent("HeavyTree", 0, 5000)
	p2 = Agents.Agent("Optimus", 1, 100)

	players = [p1, p2]

	print("Started...")

	history = []
	wins = 0
	draws = 0
	losses = 0

	num_games = 100
	percent_show = 100  # percentage of games to print status output
	player_to_track = 0

	for n in range(num_games):

		if (n + 1) % ((num_games * 100/percent_show) / num_games) == 0:
			print("Game {}".format(n + 1))

		win, draw, lose = play(ENV, p1, p2, 0, player_to_track, False)

		wins += win
		draws += draw
		losses += lose

		averages = [100*wins/(n+1), 100*draws/(n+1), 100*losses/(n+1)]
		history.append(averages)

	master = plt.figure()
	plt.plot(history)
	plt.xlabel("Number of Games")
	plt.ylabel("Percentage")
	plt.savefig("cumulative_accuracy.png")

	print("\n{} Win Percentage: {}%".format(players[player_to_track].agentType, history[-1][0]))
	print("{} Draw Percentage: {}%".format(players[player_to_track].agentType, history[-1][1]))
	print("{} Loss Percentage: {}%".format(players[player_to_track].agentType, history[-1][2]))