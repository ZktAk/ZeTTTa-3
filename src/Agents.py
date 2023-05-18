import copy
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
		if len(possibleActions) == 0:
			return None
		return random.choice(possibleActions)


class qTable_Node():
	def __init__(self, initialState, identifier):
		"""data: A Dictionary that stores all the data related to the node"""

		self.state = initialState
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

	def move(self, state):

		action = None
		actions = []

		currentState = qTable_Node(state, identifier=np.copy(state.board))

		index = locate(currentState.id, self.masterNodes)
		if index == False:
			self.masterNodes.append(currentState)
		else:
			currentState = self.masterNodes[index]

		self.gameNodes.append(currentState)

		if random.random() <= self.p:
			for act, val in currentState.actions.items():
				actions.append(act)
		else:
			best_val = float("-inf")
			for act, val in currentState.actions.items():
				if val == best_val:
					actions.append(act)
				elif val > best_val:
					best_val = val
					actions = [act]

		action = random.choice(actions)
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
			self.gameNodes[n].actions[actionTaken] += score

		self.gameNodes = []
		self.gameActions = []
		self.p *= self.gamma


class MCTS_Node():

	def __init__(self, state, parent=None, player=None, action=None):

		self.state = copy.deepcopy(state)

		self.parent = parent
		self.visits = 0
		self.winsOrDraws = 0
		self.action = action

		if player is not None:
			self.id = player
		else:
			self.id = self.parent.id * -1

		self.children = None

	def init_children(self):

		if self.children is not None: return

		actions = self.state.getPossibleActions()

		self.children = []

		for action in actions:
			child_state = copy.deepcopy(self.state).takeAction(action)
			self.children.append(MCTS_Node(child_state, self, action=action))


	def value(self):
		if self.visits == 0: return 0

		score = self.winsOrDraws
		success_rate = score / self.visits
		return success_rate


	def UCT(self, c=math.sqrt(2.0)):

		if self.visits == 0:
			return math.inf

		success_rate = self.value()
		exploration_term = c * math.sqrt(math.log(self.parent.visits) / self.visits)
		UCT = success_rate + exploration_term
		return UCT

	def exploration_rate(self, c=math.sqrt(2)):
		if self.visits == 0: return math.inf
		exploration_term = c * math.sqrt(math.log(self.parent.visits) / self.visits)
		return exploration_term

	def success_rate(self):
		success_rate = self.value()
		return success_rate


class MCTS(Agent):

	def __init__(self):
		super().__init__()
		self.nodes = {}

	def select(self, node, c=math.sqrt(2)):

		current_node = node

		while current_node.children is not None and len(current_node.children) != 0:

			best_val = float("-inf")
			nodes = []

			for child in current_node.children:
				value = child.UCT(c)
				if value > best_val:
					nodes = [child]
					best_val = value
				if value == best_val:
					nodes.append(child)

			current_node = random.choice(nodes)

		return current_node

	def expand(self, node):
		node.init_children()

		if len(node.children) == 0:
			return node

		# new_node = random.choice(node.children)
		new_node = self.select(node)
		return new_node

	def simulate(self, node):
		state = node.state
		current_node = node
		rand_agent = Random()

		while not state.isTerminal:

			action = rand_agent.move(state)
			state = state.takeAction(action)
			current_node = MCTS_Node(state, current_node)

		return current_node

	def backprop(self, winner_id, leaf_node, initial_node):
		node = leaf_node
		while node is not None:
			node.visits += 1
			if node.id == winner_id:
				node.winsOrDraws += 1
			if node is initial_node:
				break

			node = node.parent


	def think(self, initial_node, iterations=100):

		for n in range(iterations):
			parent_Node = initial_node

			selected_node = self.select(parent_Node)

			expanded_node = self.expand(selected_node)
			expanded_node = self.nodes.setdefault(str(expanded_node.state.board), expanded_node)

			winning_node = self.simulate(expanded_node)

			winning_player = winning_node.id * -1

			self.backprop(winning_player, expanded_node, initial_node)


	def pick(self, node):

		best_val = float("-inf")
		nodes = []

		for child in node.children:
			value = child.visits
			if value > best_val:
				nodes = [child]
				best_val = value
			if value == best_val:
				nodes.append(child)

		choice = random.choice(nodes)
		action = choice.action

		return action

	def move(self, state):

		parent_Node = self.nodes.setdefault(str(state.board), MCTS_Node(state, player=1))

		self.think(parent_Node, 200)
		action = self.pick(parent_Node)

		return action




