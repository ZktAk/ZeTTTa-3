import numpy as np
import random
import Environments
import Agents
import matplotlib.pyplot as plt


def play(env, players, display=False):

	current_state = env()  # reset the environment

	if display:
		current_state.print(2)

	PlayerX = random.randint(0,1)  # The player with the crosses always goes first.
	PlayerO = 1 if PlayerX == 0 else 0

	currentPlayer = PlayerX

	while True:

		action = players[currentPlayer].move(current_state)
		current_state = current_state.takeAction(action)

		currentPlayer = PlayerO if currentPlayer == PlayerX else PlayerX

		if display:
			current_state.print(1)


		if current_state.isWin():
			if display: print("{} Won!".format(current_state.symbols[action.symbol]))
			players[currentPlayer].giveReward(-1)

			if currentPlayer == 1:
				players[0].giveReward(1)
			else:
				players[1].giveReward(1)

			return

		elif current_state.isDraw():
			if display: print("Game is a Draw")

			players[0].giveReward(0)
			players[1].giveReward(0)

			return


if __name__ == '__main__':

	TicENV = Environments.TicTacToeState
	players = [Agents.QTable(0.999939), Agents.Random()]  # Agents.QTable(0.99995)

	history = []

	print("Started...")

	for n in range(100000):

		if (n+1) % (100) == 0:
			print("Game {}".format(n+1))
		play(TicENV, players, display=False)
		history.append(players[0].getPercentages())

	winPercentage = []
	drawPercentage = []
	lossPercentage = []

	for n in history:
		winPercentage.append(n[0])
		drawPercentage.append(n[1])
		lossPercentage.append(n[2])


	win = plt.figure()
	plt.plot(winPercentage)
	plt.xlabel("Number of Games")
	plt.ylabel("Win Percentage")
	plt.savefig("cumulative_wins.png")

	draw = plt.figure()
	plt.plot(drawPercentage)
	plt.xlabel("Number of Games")
	plt.ylabel("Draw Percentage")
	plt.savefig("cumulative_draws.png")

	loss = plt.figure()
	plt.plot(lossPercentage)
	plt.xlabel("Number of Games")
	plt.ylabel("Loss Percentage")
	plt.savefig("cumulative_losses.png")

	print("Win Percentage: {}%".format(history[-1][0]))
	print("Loss Percentage: {}%".format(history[-1][2]))
	print("Draw Percentage: {}%".format(history[-1][1]))

	print("unique game scenarios: {}".format(len(players[0].masterNodes)))
	for n in range(10):
		print("Move Rankings for Position {}: {}".format(n, players[0].masterNodes[n].actions.values()))
	#print("Move Rankings for Starting Position: {}".format(players[0].masterNodes[2].actions.values()))



