import random

# Global list to store used IDs
used_ids = []

class Agent:
	"""Base class for Tic-Tac-Toe agents, providing a unique identifier and placeholder methods."""

	def __init__(self):
		"""Initialize the agent with a unique identifier."""
		self.UID = self.generate_unique_id()  # Assign a unique ID to the agent

	def generate_unique_id(self):
		"""Generate a unique random ID for the agent.

        Returns:
            int: A unique 4-digit identifier not already in use.
        """
		while True:
			# Generate a random number (or any other unique identifier)
			uid = random.randint(1000, 9999)  # Generate a random 4-digit number
			if uid not in used_ids:  # Check if the ID is unused
				used_ids.append(uid)  # Add to used IDs list
				return uid

	def give_reward(self, reward):
		"""Placeholder method for handling rewards.

        Args:
            reward (float): The reward value received by the agent.

        Note:
            This method is intended to be overridden by subclasses.
        """
		pass

	def remember(self, done, reward, action, observation, prev_obs):
		"""Placeholder method for storing experience for learning.

        Args:
            done (bool): Whether the game is finished.
            reward (float): The reward received for the action.
            action (int): The action taken (move index).
            observation (list): The current game state.
            prev_obs (list): The previous game state.

        Note:
            This method is intended to be overridden by subclasses for learning purposes.
        """
		pass