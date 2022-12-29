import numpy as np
import random
import Environments
import Agents
import matplotlib.pyplot as plt


def play(env, players, display=False):

	env.__init__()  # reset the environment

	if display:
		env.print(2)

	PlayerX = random.randint(0,1)  # The player with the crosses always goes first.
	PlayerO = 1 if PlayerX == 0 else 0

	currentPlayer = PlayerX

	while True:

		move = players[currentPlayer].move(env.state)

		"""if currentPlayer == PlayerX:
			move = players[currentPlayer].move(env.state)
		else:  # currentPlayer == PlayerO
			state = np.copy(env.state)
			x = np.copy(state[0])
			state[0] = np.copy(state[1])
			state[1] = x
			move = players[currentPlayer].move(state)"""

		piece = 0 if currentPlayer == PlayerX else 1
		env.move(piece, move[0], move[1])

		currentPlayer = PlayerO if currentPlayer == PlayerX else PlayerX

		if display:
			env.print(1)

		win, draw, winner = env.win()
		if win:
			if display: print("{} Won!".format(env.symbols[winner]))
			players[currentPlayer].giveReward(1)

			if currentPlayer == 1:
				players[0].giveReward(-1)
			else:
				players[1].giveReward(-1)

			return

		elif draw:
			if display: print("Game is a Draw")

			players[0].giveReward(0)
			players[1].giveReward(0)

			return


if __name__ == '__main__':

	TicENV = Environments.TicTacToe()
	players = [Agents.QTable(0.9996), Agents.Random()]

	history = []

	numGames = 100000

	for n in range(numGames):

		if (n+1) % (100) == 0:
			print("Game {}".format(n+1))
			#print(players[0].moves[0])
		play(TicENV, players, display=False)
		history.append(players[0].getPercentages())

	#print("end of games")

	graph = []

	for n in history:
		graph.append(n[0])

	plt.plot(graph)
	plt.xlabel("Number of Games")
	plt.ylabel("Win percentage over time")
	plt.savefig("cumulative_accuracy.png")

	print("Win Percentage: {}%".format(history[-1][0]))
	print("Loss Percentage: {}%".format(history[-1][2]))
	print("Draw Percentage: {}%".format(history[-1][1]))

	print("unique game scenarios: {}".format(len(players[0].states)))
	print("Move Rankings for Starting Position: {}".format(players[0].moves[0]))



