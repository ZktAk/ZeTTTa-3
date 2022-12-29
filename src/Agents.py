import math
import random
import numpy as np


def containsElement(arr, val):
	for n in arr:
		if n == val: return True
	return False


def arrEqualsArr(arr1, arr2):

	elementsThatMatch = arr1.flatten() == arr2.flatten()

	for val in elementsThatMatch:
		if val == False: return False

	return True


def containsArray(superArr, subArr):

	for arr in superArr:
		if arrEqualsArr(arr, subArr): return True
	return False


class Agent():
	def __init__(self):
		self.wins = 0
		self.draws = 0
		self.losses = 0
		self.numGames = 0


	def giveReward(self, reward):
		self.numGames += 1
		if reward == 1:	self.wins += 1
		elif reward == 0: self.draws += 1
		elif reward == -1: self.losses += 1


	def get(self):
		return [self.wins, self.draws, self.losses]


	def getPercentages(self):
		winP = self.wins / self.numGames
		drawP = self.draws / self.numGames
		lossP = self.losses / self.numGames

		return [winP, drawP, lossP]


class Random(Agent):
	def __init__(self):
		super().__init__()

	def move(self, state):

		while True:
			row = random.randint(0, len(state[-1])-1)
			cell = random.randint(0, len(state[-1][row])-1)
			if state[-1][row][cell] == 1: return [row, cell]


class QTable(Agent):

	def __init__(self, gamma):
		super().__init__()

		self.p = 1
		self.gamma = gamma
		self.states = []
		self.moves = []

		self.gameStates = []
		self.gameMoves = []


	def move(self, state):

		move = None

		if not containsArray(self.states, state):
			self.states.append(np.copy(state))
			self.moves.append(np.zeros((3,3), int))

		if random.random() <= self.p:  # <= self.p
			#print("Selecting Random Move")
			while True:  # pick random move
				row = random.randint(0, len(state[-1])-1)
				cell = random.randint(0, len(state[-1][row])-1)
				if state[-1][row][cell] == 1:
					move = [row, cell]
					break

		else:
			#print("Selecting Best Move\n\n\n")
			stateIndex = None

			for n in range(len(self.states)):
				if arrEqualsArr(self.states[n], state):
					stateIndex = n
					break

			while True:  # pick best move

				index1D = np.argmax(self.moves[stateIndex])
				row = math.floor(index1D / 3)
				cell = index1D - (row * 3)
				if state[-1][row][cell] != 1: self.moves[stateIndex][row][cell] = self.moves[stateIndex].flatten()[self.moves[stateIndex].argmin()] - 1
				else:
					move = [row, cell]
					break

		self.gameStates.append(np.copy(state))
		self.gameMoves.append(move)
		return move


	def giveReward(self, reward):
		super().giveReward(reward)

		score = 0

		if reward == 1:
			score = 3
		elif reward == 0:
			score = 1
		elif reward == -1:
			score = reward

		for s in range(len(self.gameStates)):

			stateIndex = 0

			for n in range(len(self.states)):
				if arrEqualsArr(self.states[n], self.gameStates[s]):
					stateIndex = n
					break

			row = self.gameMoves[s][0]
			column = self.gameMoves[s][1]

			self.moves[stateIndex][row][column] += score

		self.gameStates = []
		self.gameMoves = []
		self.p *= self.gamma
