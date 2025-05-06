import copy
import random
import numpy as np
from collections import deque
from game_utils import bin_to_array, get_legal_indices
from Agents import Agent
import math

# np.random.seed(1)  # Commented out for testing/debugging purposes

def normalize(arr):
	"""Normalize an array to sum to 1.

	Args:
	    arr (np.ndarray): Input array to normalize.

	Returns:
	    np.ndarray: Normalized array where values sum to 1.
	"""
	sum = np.sum(arr)
	prob = arr / sum
	return prob

def relu(mat):
	"""Apply ReLU activation function to an array.

    Args:
        mat (np.ndarray): Input array.

    Returns:
        np.ndarray: Array with ReLU applied (negative values set to 0).
    """
	return np.multiply(mat, (mat > 0))

def relu_derivative(mat):
	"""Compute the derivative of the ReLU activation function.

    Args:
        mat (np.ndarray): Input array.

    Returns:
        np.ndarray: Array where elements are 1 if input > 0, else 0.
    """
	return (mat > 0) * 1

def sigmoid(mat):
	"""Apply sigmoid activation function to an array.

    Args:
        mat (np.ndarray): Input array.

    Returns:
        np.ndarray: Array with sigmoid applied.
    """
	return 1 / (1 + pow(math.e, -mat))

class NNLayer:
	"""A class representing a single layer in a neural network."""

	def __init__(self, input_size, output_size, activation=None, lr=0.001):
		"""Initialize a neural network layer.

        Args:
            input_size (int): Number of input nodes (including bias).
            output_size (int): Number of output nodes.
            activation (callable, optional): Activation function (e.g., ReLU).
            lr (float): Learning rate for weight updates.
        """
		self.input_size = input_size
		self.output_size = output_size
		self.weights = np.random.uniform(low=-0.5,
		                                 high=0.5,
		                                 size=(input_size,
		                                 output_size))      # Random weight initialization
		self.stored_weights = np.copy(self.weights)         # Copy of weights for forward pass without backprop
		self.activation_function = activation               # Activation function
		self.lr = lr                                        # Learning rate
		self.m = np.zeros((input_size, output_size))        # First moment for Adam optimizer
		self.v = np.zeros((input_size, output_size))        # Second moment for Adam optimizer
		self.beta_1 = 0.9                                   # Adam beta1 parameter
		self.beta_2 = 0.999                                 # Adam beta2 parameter
		self.time = 1                                       # Time step for Adam optimizer
		self.adam_epsilon = 0.00000001                      # Small constant to prevent division by zero


	def forward(self, inputs, remember_for_backprop=True):
		"""Compute the forward pass for this layer.

        Args:
            inputs (np.ndarray): Input array of shape (batch_size, input_size).
            remember_for_backprop (bool): If True, store intermediate values for backpropagation.

        Returns:
            np.ndarray: Output after applying weights and activation function.
        """
		input_with_bias = np.append(inputs, 1)  # Add bias term
		unactivated = None

		# Compute weighted sum using current or stored weights
		if remember_for_backprop:
			unactivated = np.dot(input_with_bias, self.weights)
		else:
			unactivated = np.dot(input_with_bias, self.stored_weights)

		output = unactivated
		if self.activation_function != None:  # assuming here the activation function is relu, this can be made more robust
			output = self.activation_function(output)  # Apply activation function

		# Store values for backpropagation
		if remember_for_backprop:
			self.backward_store_in = input_with_bias
			self.backward_store_out = np.copy(unactivated)

		return output


	def update_weights(self, gradient):
		"""Update layer weights using the Adam optimizer.

        Args:
            gradient (np.ndarray): Gradient of the loss with respect to the weights.
        """
		m_temp = np.copy(self.m)
		v_temp = np.copy(self.v)

		# Update first and second moments
		m_temp = self.beta_1 * m_temp + (1 - self.beta_1) * gradient
		v_temp = self.beta_2 * v_temp + (1 - self.beta_2) * (gradient * gradient)

		# Bias-corrected moments
		m_vec_hat = m_temp / (1 - np.power(self.beta_1, self.time + 0.1))
		v_vec_hat = v_temp / (1 - np.power(self.beta_2, self.time + 0.1))

		# Update weights using Adam formula
		self.weights -= np.divide(self.lr * m_vec_hat, np.sqrt(v_vec_hat) + self.adam_epsilon)

		self.m = np.copy(m_temp)
		self.v = np.copy(v_temp)


	def update_time(self):
		"""Increment the time step for the Adam optimizer."""
		self.time += 1


	def update_stored_weights(self):
		"""Update the stored weights to match the current weights."""
		self.stored_weights = np.copy(self.weights)


	def backward(self, gradient_from_above):
		"""Compute the backward pass for this layer.

        Args:
            gradient_from_above (np.ndarray): Gradient from the layer above.

        Returns:
            np.ndarray: Gradient to pass to the previous layer.
        """
		adjusted_mul = gradient_from_above

		# Apply activation derivative if applicable. Possibly pointless.
		if self.activation_function is not None:
			adjusted_mul = np.multiply(relu_derivative(self.backward_store_out), gradient_from_above)

		# Compute weight gradient and update weights
		D_i = np.dot(np.transpose(np.reshape(self.backward_store_in, (1, len(self.backward_store_in)))), np.reshape(adjusted_mul, (1, len(adjusted_mul))))
		delta_i = np.dot(adjusted_mul, np.transpose(self.weights))[:-1]  # Exclude bias term
		self.update_weights(D_i)

		return delta_i

class RLAgent(Agent):
	"""A class representing a reinforcement learning agent using a neural network for Tic-Tac-Toe."""

	def __init__(self, input_size, num_hidden, hidden_size, output_size, gamma=0.8, epsilon_decay=0.997,
	             epsilon_min=0.01, batch_size=16):
		"""Initialize the reinforcement learning agent.

        Args:
            input_size (int): Size of the input layer (27 for Tic-Tac-Toe bitboards).
            num_hidden (int): Number of hidden layers.
            hidden_size (int): Size of each hidden layer.
            output_size (int): Size of the output layer (9 for Tic-Tac-Toe moves).
            gamma (float): Discount factor for future rewards.
            epsilon_decay (float): Decay rate for exploration probability.
            epsilon_min (float): Minimum exploration probability.
            batch_size (int): Number of samples for experience replay.
        """
		super().__init__()  # Call the parent class (Agent) initializer
		self.agentType = "NeuralNetwork"  # Set the agent type identifier

		self.input_size = input_size
		self.num_hidden = num_hidden
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.gamma = gamma                  # Discount factor
		self.epsilon_decay = epsilon_decay  # Exploration decay rate
		self.epsilon_min = epsilon_min      # Minimum exploration probability
		self.batch_size = batch_size        # Batch size for training
		self.epsilon = 1.0                  # Initial exploration probability

		self.current_piece = 0              # Current player's piece (0 for x, 1 for o)
		self.memory = deque([], 1000000)    # Memory buffer for experience replay

		# Initialize neural network layers
		self.layers = [NNLayer(self.input_size + 1, self.hidden_size, activation=relu)]
		for i in range(self.num_hidden - 1):
			self.layers.append(NNLayer(self.hidden_size + 1, self.hidden_size, activation=relu))
		self.layers.append(NNLayer(self.hidden_size + 1, self.output_size))


	def get_observation(self, observation):
		"""Convert game observation to neural network input.

        Args:
            observation (list): Game state containing [bitboards, piece, move_number, env].

        Returns:
            np.ndarray: Array of size 27 representing the game state.
        """
		bitboards, piece, move_number, _ = observation
		x_bitboard, y_bitboard, empty = bitboards
		# Combine bitboards into a single value, adjusting for player perspective
		if piece == 1:
			bitboard = y_bitboard << 18 | x_bitboard << 9 | empty
		else:
			bitboard = x_bitboard << 18 | y_bitboard << 9 | empty
		#bitboard = x_bitboard << 9 | y_bitboard  # Commented out for testing/debugging purposes
		input_layer = bin_to_array(bitboard, 27)  # Convert to array
		return input_layer

	def move(self, observation):
		"""Select a move using the neural network or random exploration.

        Args:
            observation (list): Game state containing [bitboards, piece, move_number, env].

        Returns:
            int: The index (0-8) of the chosen move.
        """
		bitboards, piece, move_number, env = observation
		x, y, empty = bitboards
		obs = self.get_observation(copy.deepcopy(observation))
		empty_array = bin_to_array(empty, 9)  # Array indicating empty squares

		values = self.forward(obs) # Get move probabilities

		# Following code commented out for testing/debugging purposes
		#print(f"Previous Values: {prev_values} \nEmpty Array: {empty_array} \nNew Values: {values}")
		#print(len(self.memory))

		# Choose move: exploit (best move) or explore (random move)
		if (np.random.random() > self.epsilon):
			move = np.argmax(values)
			#print(f"Best Move: {values[move]}\n")  # Commented out for testing/debugging purposes
		else:
			move = random.choice(get_legal_indices(bitboards))
			#print(f"Random Move: {move}\n")  # Commented out for testing/debugging purposes
		return move


	def forward(self, observation, remember_for_backprop=True):
		"""Compute the forward pass through the neural network.

        Args:
            observation (np.ndarray): Input observation array.
            remember_for_backprop (bool): If True, store values for backpropagation.

        Returns:
            np.ndarray: Probabilities for each move, adjusted for legal moves.
        """
		empty_array = observation[0:9]  # Extract empty squares
		#print(empty_array)  # Commented out for testing/debugging purposes

		vals = np.copy(observation)

		for layer in self.layers:
			vals = layer.forward(vals, remember_for_backprop)  # Pass through each layer

		vals = np.array(vals)
		sum = np.sum(vals)
		prob = vals / sum  # Normalize outputs

		# Apply sigmoid and mask illegal moves
		for n in range(len(prob)):
			prob[n] = sigmoid(prob[n])
			if empty_array[n] == 0:
				prob[n] = 0

		prob = prob + [val/100 for val in empty_array]  # Slightly favor empty squares
		return prob


	def remember(self, done, reward, action, observation, prev_observation):
		"""Store an experience in memory for experience replay.

        Args:
            done (bool): Whether the game is finished.
            reward (float): Reward received for the move.
            action (int): Action taken (move index).
            observation (list): Current game state.
            prev_observation (list): Previous game state.
        """
		obs = self.get_observation(observation)
		prev_obs = self.get_observation(prev_observation)
		self.memory.append([done, reward, action, obs, prev_obs, prev_observation[-2]])  # Store experience
		self.experience_replay()

	def experience_replay(self):
		"""Perform experience replay to train the neural network."""
		if (len(self.memory) < 1000):  # Wait until enough experiences are collected
			return

		# Sample a batch of experiences
		batch_indices = np.random.choice(len(self.memory), self.batch_size)

		for index in batch_indices:
			done, reward, action_selected, new_obs, prev_obs, move_num = self.memory[index]
			action_values = self.forward(prev_obs, remember_for_backprop=True)  # Predicted Q-values
			next_action_values = self.forward(new_obs, remember_for_backprop=False)  # Next state Q-values
			experimental_values = np.copy(action_values)

			# Update Q-value for the selected action
			experimental_values[action_selected] = reward + 0.5 + self.gamma * np.max(next_action_values)

			# The following commented-out code is for testing/debugging purposes
			# print(reward)
			#print("{}: {}".format(reward, experimental_values[action_selected]))
			"""if done:
				experimental_values[action_selected] = reward+1 #math.ceil(reward)
				#print(reward, experimental_values[action_selected])
			else:
				n = 0.5 if move_num < 5 else 1
				experimental_values[action_selected] = 0.5 + self.gamma * np.max(next_action_values)
				#print(experimental_values[action_selected])
			"""
			self.backward(action_values, experimental_values)  # Backpropagate error

		# Decay exploration probability
		self.epsilon = self.epsilon if self.epsilon < self.epsilon_min else self.epsilon * self.epsilon_decay

		# Update layer time steps and stored weights
		for layer in self.layers:
			layer.update_time()
			layer.update_stored_weights()


	def backward(self, calculated_values, experimental_values):
		"""Perform backpropagation to update neural network weights.

        Args:
            calculated_values (np.ndarray): Predicted Q-values.
            experimental_values (np.ndarray): Target Q-values based on rewards and next state.
        """
		delta = (calculated_values - experimental_values)  # Compute error
		for layer in reversed(self.layers):
			delta = layer.backward(delta)  # Backpropagate error through layers