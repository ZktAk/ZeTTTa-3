import random

# Global list to store used IDs
used_ids = []

class Agent:

	def __init__(self):

		self.UID = self.generate_unique_id()

	def generate_unique_id(self):
		while True:
			# Generate a random number (or any other unique identifier)
			uid = random.randint(1000, 9999)
			if uid not in used_ids:
				used_ids.append(uid)
				return uid

	def give_reward(self, reward):
		pass

	def remember(self, done, reward, action, observation, prev_obs):
		pass

