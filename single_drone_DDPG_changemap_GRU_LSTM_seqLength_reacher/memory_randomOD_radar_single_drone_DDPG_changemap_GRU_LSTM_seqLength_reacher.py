from collections import namedtuple
import random
Experience = namedtuple('Experience',
						('states','actions','act_logprob','next_states','rewards','dones','history_info','cur_hidden','next_hidden', 'lstm_hidden', 'gru_hidden'))

class ReplayMemory:
	def __init__(self,capacity, history_length):
		self.capacity = capacity
		self.memory = []
		self.position = 0
		self.history_seq_length = history_length
		self.sampling_indexes = None
		
	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)

		self.memory[self.position] = Experience(*args)
		self.position = int((self.position + 1)%self.capacity)
		
	def sample(self, batch_size, random_sample=True):
		if random_sample:
		# print(len(self.memory),batch_size)
			return random.sample(self.memory, batch_size)
		else:
			# Return the first `batch_size` elements in order
			return self.memory[:batch_size]

	def sample_by_index(self, indexes):
		self.sampling_indexes = indexes
		return [self.memory[i] for i in self.sampling_indexes]
	
	def __len__(self):
		return len(self.memory)

	def clear(self):
		"""Clears the replay memory and resets position and any auxiliary indexes."""
		self.memory.clear()  # clear all stored experiences
		self.position = 0  # reset the insertion position
		self.sampling_indexes = None  # clear any sampling indexes