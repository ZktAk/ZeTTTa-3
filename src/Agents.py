import math
import random
import numpy as np
from Environments import TicTacToeState


def containsElement(arr, val):
	for n in arr:
		if n == val: return True
	return False


def arrEqualsArr(arr1, arr2):

	#print("arr1: {}".format(arr1.flatten()))
	#print("arr2: {}".format(arr2.flatten()))

	elementsThatMatch = arr1.flatten() == arr2.flatten()

	for val in elementsThatMatch:
		if val == False:
			#print("False\n")
			return False
	#print("True\n")
	return True


def containsArray(superArr, subArr):

	for arr in superArr:
		if arrEqualsArr(arr, subArr): return True
	return False


def to2DIndex(index1D, shape=(3,3)):

	row = math.floor(index1D / shape[0])
	cell = index1D - (row * shape[1])

	return row, cell




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

	def move(self, initialState):
		possibleActions = initialState.getPossibleActions()
		return random.choice(possibleActions)


class node():
	def __init__(self, initialState, data, identifier):
		"""data: A Dictionary that stores all the data related to the node"""

		self.state = initialState
		self.dat = data
		self.id = identifier

		self.PossibleActions = self.state.getPossibleActions()

		self.actions = {}

		for PossibleAction in self.PossibleActions:
			self.actions[PossibleAction] = 0


def locate(identifier, arr):
	"""Returns the index of the Node with the provided identifier in the provided array if found, else returns False."""

	for n in range(len(arr)):
		if arrEqualsArr(arr[n].id, identifier):  # if arr[n].id == identifier:
			return n
	return False


class QTable(Agent):

	def __init__(self, gamma):
		super().__init__()

		self.p = 1
		self.gamma = gamma

		self.masterNodes = []
		self.gameNodes = []
		self.gameActions = []

		"""self.states = []
		self.moves = []

		self.gameStates = []
		self.gameMoves = []"""


	def move(self, state):

		currentState = node(state, {"score": 0}, identifier=np.copy(state.board))

		index = locate(currentState.id, self.masterNodes)
		if index == False:
			self.masterNodes.append(currentState)
		else:
			currentState = self.masterNodes[index]

		self.gameNodes.append(currentState)


		if random.random() <= self.p:  # <= self.p
			keys = []
			for key, val in currentState.actions.items():
				keys.append(key)
			action = random.choice(keys)
			self.gameActions.append(action)
			return action

		else:
			best_val = float("-inf")
			best_actions = []
			for act, val in currentState.actions.items():
				if val == best_val:
					best_actions.append(act)
				elif val > best_val:
					best_val = val
					best_actions = [act]

			action = random.choice(best_actions)
			self.gameActions.append(action)
			return action


	def giveReward(self, reward):
		super().giveReward(reward)

		score = 0

		if reward == 1:
			score = 3
		elif reward == 0:
			score = 1
		elif reward == -1:
			score = reward

		for n in range(len(self.gameNodes)):
			actionTaken = self.gameActions[n]
			#print("before: {}".format(self.gameNodes[n].actions[actionTaken]))
			self.gameNodes[n].actions[actionTaken] += score
			#print("after: {}\n".format(self.gameNodes[n].actions[actionTaken]))

		"""for s in range(len(self.gameStates)):

			stateIndex = 0

			for n in range(len(self.states)):
				if arrEqualsArr(self.states[n], self.gameStates[s]):
					stateIndex = n
					break

			row = self.gameMoves[s][0]
			column = self.gameMoves[s][1]

			self.moves[stateIndex][row][column] += score"""

		self.gameNodes = []
		self.gameActions = []
		self.p *= self.gamma


class Node():
	def __init__(self, env, state=None, row=None, column=None, parent=None, id=1):
		self.n = 0  # represents the number of times the node has been considered (visit count)
		self.w = 0  # represents the number of wins considered for that node

		self.environment = env

		self.symbol = id
		self.state = state
		self.parent = parent

		self.row = row
		self.column = column

		self.expanded = False


		if parent is not None:
			self.stateENV = env(state=np.copy(parent.board))
			self.stateENV.move([1, -1].index(self.symbol), row, column)
			self.state = self.stateENV.board
		else:
			self.stateENV = env(state=state)
			self.state = self.stateENV.board

		win, draw, winner = self.stateENV.isWin()
		self.isTerminal = win or draw


		self.childNodes = []


	def generate(self):

		#print(self.stateENV.state)

		debug = ""

		for n in range(9):
			row, column = to2DIndex(n, (3, 3))

			temp = str([row, column])

			temp += " is Legal: {}".format(self.stateENV.legal(row, column))

			debug += "\n" + temp

			if self.stateENV.legal(row, column):
				self.childNodes.append(Node(env=self.environment, row=row, column=column, parent=self, id=self.symbol*-1))

		if len(self.childNodes)==0:
			print(self.stateENV.board)
			print(debug)

		self.expanded = True


class MCTS(Agent):
	def __init__(self, env=TicTacToeState):
		super().__init__()
		self.ENV = env


	def getNode(self, state):
		pass


	def select(self, node, c=math.sqrt(2)):

		best_val = float("-inf")
		children = []

		#print(node.childNodes)

		for childNode in node.childNodes:

			"""
			row, column = to2DIndex(node.childNodes.index(childNode))
			print([row, column])
			"""

			UCB1 = math.inf
			if childNode.n != 0:
				winrate = childNode.w / childNode.n
				explorationTerm = c * math.sqrt(math.log(node.n) / childNode.n)
				UCB1 = winrate + explorationTerm

			if UCB1 == best_val:
				children.append(childNode)
			elif UCB1 > best_val:
				best_val = UCB1
				children = [childNode]

		return random.choice(children)


	def expand(self, node):
		node.generate()

		#print(node.childNodes)
		leaf = random.choice(node.childNodes)


		return leaf


	def simulate(self, node):

		leaf = node

		if not leaf.isTerminal:
			if not leaf.expanded:
				leaf.generate()

			leaf = random.choice(node.childNodes)

		return leaf  # in addition to returning the terminal leaf, we must also return wheither that leaf won, lost, or drawed because we need to know if we should increment the win count of the Root Node


	@staticmethod
	def backpropagate(root, end):

		node = end

		while node.parent is not None:
			node.n += 1
			if node.symbol == end.symbol:
				node.w += 1
			node = node.parent

		root.n += 1

		if root.symbol == end.symbol:
				node.w += 1


	@staticmethod
	def selectBest(node):
		best = [0]
		children = []

		# print(node.childNodes)

		for childNode in node.childNodes:

			row, column = to2DIndex(node.childNodes.index(childNode))

			# print([row, column])

			winrate = math.inf
			if childNode.n != 0:
				winrate = childNode.w / childNode.n


			if winrate == best[0]:
				best.append(winrate)
				children.append(childNode)
			elif winrate > best[0]:
				best = [winrate]
				children = [childNode]

		return random.choice(children)

	def move(self, state, piece):

		rootSymbol = [1, -1][piece]
		rootNode = Node(state=state, env=self.ENV, id=rootSymbol)
		rootNode.generate()

		for n in range(100):

			firstLeaf = self.select(rootNode)  # this returns a child node of the Root Node based on UCB1

			if not firstLeaf.isTerminal:

				secondLeaf = self.expand(firstLeaf)

				terminalLeaf = self.simulate(secondLeaf)

				self.backpropagate(rootNode, terminalLeaf)


		choice = self.selectBest(rootNode)

		row = choice.row
		column = choice.column

		return [row, column]




