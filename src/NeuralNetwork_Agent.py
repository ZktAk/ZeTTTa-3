import copy
import random
import numpy as np
from collections import deque
from BinHelp import bin_to_array, get_legal_indices
from Agents import Agent
import math

#np.random.seed(1)

def normalize(arr):
	sum = np.sum(arr)
	prob = arr / sum
	return prob

def relu(mat):
	return np.multiply(mat, (mat > 0))

def relu_derivative(mat):
	return (mat > 0) * 1

def sigmoid(mat):
	return 1 / (1 + pow(math.e, -mat))

class NNLayer:
	# class representing a neural net layer
	def __init__(self, input_size, output_size, activation=None, lr=0.001):
		self.input_size = input_size
		self.output_size = output_size
		self.weights = np.random.uniform(low=-0.5, high=0.5, size=(input_size, output_size))
		self.stored_weights = np.copy(self.weights)
		self.activation_function = activation
		self.lr = lr
		self.m = np.zeros((input_size, output_size))
		self.v = np.zeros((input_size, output_size))
		self.beta_1 = 0.9
		self.beta_2 = 0.999
		self.time = 1
		self.adam_epsilon = 0.00000001

	# Compute the forward pass for this layer
	def forward(self, inputs, remember_for_backprop=True):
		# inputs has shape batch_size x layer_input_size

		input_with_bias = np.append(inputs, 1)
		unactivated = None

		if remember_for_backprop:
			unactivated = np.dot(input_with_bias, self.weights)
		else:
			unactivated = np.dot(input_with_bias, self.stored_weights)

		# store variables for backward pass
		output = unactivated

		if self.activation_function != None:  # assuming here the activation function is relu, this can be made more robust

			output = self.activation_function(output)

		if remember_for_backprop:
			self.backward_store_in = input_with_bias
			self.backward_store_out = np.copy(unactivated)

		return output


	def update_weights(self, gradient):
		m_temp = np.copy(self.m)
		v_temp = np.copy(self.v)

		m_temp = self.beta_1 * m_temp + (1 - self.beta_1) * gradient
		v_temp = self.beta_2 * v_temp + (1 - self.beta_2) * (gradient * gradient)

		m_vec_hat = m_temp / (1 - np.power(self.beta_1, self.time + 0.1))
		v_vec_hat = v_temp / (1 - np.power(self.beta_2, self.time + 0.1))

		self.weights -= np.divide(self.lr * m_vec_hat, np.sqrt(v_vec_hat) + self.adam_epsilon)

		self.m = np.copy(m_temp)
		self.v = np.copy(v_temp)


	def update_time(self):
		self.time += 1


	def update_stored_weights(self):
		self.stored_weights = np.copy(self.weights)


	def backward(self, gradient_from_above):

		adjusted_mul = gradient_from_above

		# this is pointwise
		if self.activation_function != None:
			adjusted_mul = np.multiply(relu_derivative(self.backward_store_out), gradient_from_above)

		D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))), np.reshape(adjusted_mul, (1, len(adjusted_mul))))
		delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]
		self.update_weights(D_i)

		return delta_i

class RLAgent(Agent):
	# class representing a reinforcement learning agent

	def __init__(self, input_size, num_hidden, hidden_size, output_size, gamma=0.8, epsilon_decay=0.997,
	             epsilon_min=0.01, batch_size=16):
		super().__init__()
		self.agentType = "NeuralNetwork"

		self.input_size = input_size
		self.num_hidden = num_hidden
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.gamma = gamma
		self.epsilon_decay = epsilon_decay
		self.epsilon_min = epsilon_min
		self.batch_size = batch_size
		self.epsilon = 1.0

		self.current_piece = 0

		self.memory = deque([], 1000000)

		self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu)]

		for i in range(self.num_hidden - 1):
			self.layers.append(NNLayer(self.hidden_size + 1, self.hidden_size, activation=relu))

		self.layers.append(NNLayer(self.hidden_size + 1, self.output_size))


	def get_observation(self, observation):
		bitboards, piece, move_number, _ = observation
		x_bitboard, y_bitboard, empty = bitboards
		if piece == 1:
			bitboard = y_bitboard << 18 | x_bitboard << 9 | empty
		else:
			bitboard = x_bitboard << 18 | y_bitboard << 9 | empty
		#bitboard = x_bitboard << 9 | y_bitboard
		input_layer = bin_to_array(bitboard, 27)
		return input_layer

	def move(self, observation):

		bitboards, piece, move_number, env = observation
		x, y, empty = bitboards
		obs = self.get_observation(copy.deepcopy(observation))
		empty_array = bin_to_array(empty, 9)

		values = self.forward(obs)
		#print(f"Previous Values: {prev_values} \nEmpty Array: {empty_array} \nNew Values: {values}")

		#print(len(self.memory))

		if (np.random.random() > self.epsilon):
			move = np.argmax(values)
			#print(f"Best Move: {values[move]}\n")
		else:
			move = random.choice(get_legal_indices(bitboards))
			#print(f"Random Move: {move}\n")
		return move


	def forward(self, observation, remember_for_backprop=True):

		empty_array = observation[0:9]
		#print(empty_array)

		vals = np.copy(observation)
		index = 0

		for layer in self.layers:
			vals = layer.forward(vals, remember_for_backprop)
			index = index + 1

		vals = np.array(vals)
		sum = np.sum(vals)
		prob = vals / sum

		for n in range(len(prob)):
			prob[n] = sigmoid(prob[n])
			if empty_array[n] == 0:
				prob[n] = 0

		prob = prob + [val/100 for val in empty_array]
		return prob


	def remember(self, done, reward, action, observation, prev_observation):
		obs = self.get_observation(observation)
		prev_obs = self.get_observation(prev_observation)

		self.memory.append([done, reward, action, obs, prev_obs, prev_observation[-2]])
		self.experience_replay()


	def experience_replay(self):

		if (len(self.memory) < 1000):#self.batch_size):
			return
		else:
			batch_indices = np.random.choice(len(self.memory), self.batch_size)

			for index in batch_indices:
				done, reward, action_selected, new_obs, prev_obs, move_num = self.memory[index]
				action_values = self.forward(prev_obs, remember_for_backprop=True)
				next_action_values = self.forward(new_obs, remember_for_backprop=False)
				experimental_values = np.copy(action_values)

				#print(reward)

				experimental_values[action_selected] = reward + 0.5 + self.gamma * np.max(next_action_values)

				#print("{}: {}".format(reward, experimental_values[action_selected]))

				"""if done:
					experimental_values[action_selected] = reward+1 #math.ceil(reward)
					#print(reward, experimental_values[action_selected])
				else:
					n = 0.5 if move_num < 5 else 1
					experimental_values[action_selected] = 0.5 + self.gamma * np.max(next_action_values)
					#print(experimental_values[action_selected])
					"""
				self.backward(action_values, experimental_values)

		self.epsilon = self.epsilon if self.epsilon < self.epsilon_min else self.epsilon * self.epsilon_decay

		for layer in self.layers:
			layer.update_time()
			layer.update_stored_weights()


	def backward(self, calculated_values, experimental_values):
		# values are batched = batch_size x output_size
		delta = (calculated_values - experimental_values)
		for layer in reversed(self.layers):
			delta = layer.backward(delta)