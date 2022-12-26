import random
import numpy

class Random():
	def __init__(self):
		pass

	def move(self, state):

		while True:
			row = random.randint(0, len(state[-1]) - 1)
			cell = random.randint(0, len(state[-1][row]) - 1)
			if state[-1][row][cell] == 1: return [row, cell]
