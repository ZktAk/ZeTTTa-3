import math
import numpy as np
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


class Edge():
	def __init__(self, prior_prob):
		self.n = 0           # visit count
		self.p = prior_prob  # prior probability
		self.w = 0           # intermediary value
		self.q = 0           # action-value


class Node():
	def __init__(self, parent=None, ID=1, move=None, state=np.zeros([3,3], int), prior_prob=None):

		self.player_ID = ID
		self.parent = parent
		self.board = state
		self.move_ID = move

		self.state_value = 0
		self.children = []

		if parent is not None:
			self.edge = Edge(prior_prob)

	def expand(self, priors, state_val):

		self.state_value = state_val

		for n in range(9):
			child_board = self.board.copy().reshape(9)
			child_board[n] = self.player_ID * -1
			child_board = child_board.reshape(3,3)

			move_index = np.unravel_index(n, self.board.shape)

			#print("child_board: " + str(child_board))

			self.children.append(Node(self, ID=self.player_ID * -1, move=move_index, state=child_board, prior_prob=priors[n]))


	def string(self):
		return ("ID: " + str(self.player_ID) + " | Board: " + str(self.board))

class MCTS_NN_Agent():

	def __init__(self):
		model = Sequential()

		model.add(Conv2D(10, (2,2), padding="same", input_shape=(3,3,1), activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

		model.add(Conv2D(64, (2,2), padding="same", activation='relu'))
		model.add(MaxPooling2D(pool_size=(2,2), padding="same"))

		model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
		model.add(Dense(5))

		model.add(Dense(10))
		model.add(Activation('sigmoid'))

		model.compile(loss='binary_crossentropy',
		              optimizer='adam',
		              metrics=['accuracy'])

		self.CNN = model

		self.temp_memory_targets = []
		self.memory_targets = []
		self.memory_inputs = []




	def train_Model(self):

		self.memory_inputs = np.array(self.memory_inputs)
		self.memory_targets = np.array(self.memory_targets)

		#print("\n\nInput Size: " + str(np.array(self.memory_inputs).shape))
		#print("Target Size: " + str(np.array(self.memory_targets).shape))

		self.CNN.fit(x=self.memory_inputs, y=self.memory_targets)

	def select(self, root, c=0.5):

		node = root
		values = []

		#print("root children: " + str(root.children))

		while len(node.children) != 0:  # a node without children is a leaf node

			values = []
			sumN = 0

			#print("children: \n" + str(node.children))

			for child in node.children:
				sumN += child.edge.n

			for child in node.children:
				val = child.edge.q + (c * child.edge.p * math.sqrt(sumN) / (1 + child.edge.n))
				values.append(val)

			#print("values :\n" + str(values))

			values = np.array(values).reshape((3,3))
			values[node.board != 0] = 0  # no illegal moves
			values = values.reshape(9)

			sum = np.sum(values)
			newVals = []

			for val in values:
				newVal = val
				if sum > 0.0:
					newVal = newVal / sum
				newVals.append(newVal)

			values = np.array(newVals)

			#print("sum after: " + str(np.sum(values)))

			if (node == root and np.sum(values) == 0.0):
				dirichlet_noise = np.random.dirichlet((0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3), size=1)
				values = dirichlet_noise[0]

			elif (node == root):
				dirichlet_noise = np.random.dirichlet((0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3), size=1)
				#print("dirichlet_noise: " + str(dirichlet_noise))
				values = values * dirichlet_noise[0]





			try:
				node = node.children[np.random.choice(np.argmax(values))]
			except:
				node = node.children[np.random.choice(9)]

		#best_move = np.unravel_index(np.argmax(values), node.board.shape)

		#print("node children: " + str(node.children))

		return node #best_move


	def update(self, leaf):
		node = leaf

		while node.parent is not None:
			node.edge.n += 1

			#print(node)

			if node.player_ID != leaf.player_ID:  # in actuality, node.parent owns node.edge
				node.edge.w += leaf.state_value
			else:
				node.edge.w -= leaf.state_value

			node.edge.q = node.edge.w / node.edge.n

			node = node.parent


	def choose(self, node, t=1/math.sqrt(2)):

		probabilities = []
		sumN = 0

		for child in node.children:
			sumN += pow(child.edge.n, (1/t))

		for child in node.children:
			val = pow(child.edge.n, (1/t)) / sumN
			probabilities.append(val + 0.0001)

		probabilities = np.array(probabilities)

		#print("\nprobabilities: " + str(probabilities))
		#print("sum: " + str(np.sum(probabilities)))

		for n in range(len(probabilities)):
			if node.board.reshape(9)[n] != 0:  # no illegal moves
				probabilities[n] = 0

		#print("node.board: " + str(node.board.reshape(9)))

		#print("Legal probabilities: " + str(probabilities))

		sumP = probabilities.sum()

		newProbs = []


		if sumP > 0.0:
			for prob in probabilities:
				newProb = prob
				newProb /= sumP
				newProbs.append(newProb)

			probabilities = np.array(newProbs)

		#print("adjusted probabilities: " + str(probabilities))
		#print("adjusted sum: " + str(np.sum(probabilities)))


		choice = np.random.choice(np.arange(0, 9), p=probabilities)

		new_node = node.children[choice]
		#best_move = np.unravel_index(choice, node.board.shape)

		return new_node, probabilities   # , best_move


	def remember(self, input, target_policy, target_val=0):

		#print("\ninputs: " + str(input))
		#print("")

		self.memory_inputs.append(input)

		y = np.append(target_policy, 0).reshape(10)  # append a place-holder for target_val

		#print("Targets: " + str(y))
		#print("\n")


		self.temp_memory_targets.append(y)

	def gameEnd(self, reward):

		#print("Saved Target Policy: " + str(np.array(self.temp_memory_targets).shape))

		r = reward * -1  # reward is either 1, -1, or 0

		for memory in self.temp_memory_targets:

			r *= -1
			memory[-1] = r
			#print("memory: " + str(memory))


		self.memory_targets.extend(self.temp_memory_targets)

		self.temp_memory_targets = []

		#print("Saved Target Policy: " + str(np.array(self.memory_targets).shape))

	def predict(self, node):

		state = np.copy(node.board)

		new_state = []

		#print("state: \n" + str(state))

		#state[1][1] = 72

		for row in range(len(state)):
			new_state.append([])
			for val in range(len(state[row])):

				v = state[row][val]
				v = [v]
				#print("v = " + str(v))
				new_state[row].append(v)

		new_state = np.array(new_state)

		#print("\nstate after: \n" + str(new_state))
		#print("\nstate shape: " + str(new_state.shape))

		return self.CNN.predict(new_state, verbose=0)[-1]



	def move(self, root, think_time=25):


		for n in range(think_time):

			#print("\tstep {} or {}".format(n+1, think_time))

			node = self.select(root)

			out = self.predict(node)
			#print("output: \n" + str(out))
			prior_probabilities = out[:-1]
			state_value = out[-1]

			node.expand(prior_probabilities, state_value)

			self.update(node)


		choice, target_policy = self.choose(root)
		self.remember(root.board.reshape(3,3), target_policy)
		return choice

