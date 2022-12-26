import numpy as np
import random
import Environments
import Agents


def play(env, players, displayBoard=False):

	env.__init__()  # reset the environment

	if displayBoard:
		env.print(2)

	PlayerX = random.randint(0,1)  # The player with the crosses always goes first.
	PlayerO = 1 if PlayerX == 0 else 0

	currentPlayer = PlayerX

	while True:
		move = players[currentPlayer].move(env.state)

		piece = 0 if currentPlayer == PlayerX else 1
		env.move(piece, move[0], move[1])

		currentPlayer = PlayerO if currentPlayer == PlayerX else PlayerX

		if displayBoard:
			env.print(1)

		win, draw, winner = env.win()
		if win:
			print("{} Won!".format(env.symbols[winner]))
			return
		elif draw:
			print("Game is a Draw")
			return


if __name__ == '__main__':

	TicENV = Environments.TicTacToe()
	players = [Agents.Random(), Agents.Random()]

	play(TicENV, players, displayBoard=True)
