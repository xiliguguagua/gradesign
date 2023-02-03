import torch
import torch.nn as nn
import copy.deepcopy as deepcopy

from network import emnist_net, cifar10_net

class User(object):
	"""docstring for User"""
	def __init__(self, uid):
		super(User, self).__init__()
		self.id = uid
		self.lr = lr
		self.ismalice = is_m
		self.max_iteration = max_it
		self.min_iteration = min_it

		self.local_database = ld
		self.CELoss = nn.CrossEntropyLoss()
		if task == 'emnist':
			self.local_model = emnist_net()
		else:
			self.local_model = cifar10_net()


	def local_train(self):
		optm = torch.optim.SGD(self.local_model.parameters(), lr=self.lr)
		best_loss = 1e10

		train_data, train_label, valid_data, valid_label = None, None, None, None
		for _ in range(self.max_iteration):
			self.local_model.train()
			output = self.local_model()
			ce_loss = self.CrossEntropyLoss(train_label, output)
			optm.zero_grad()
			ce_loss.backward()
			optm.step()

			self.local_model.eval()
			valid_out = self.local_model()	
			valid_loss = self.CrossEntropyLoss(valid_label, valid_out)
			if valid_loss > best_loss and _ > self.min_iteration:
				self.local_model = self.best_model
			if valid_loss < best_loss:
				best_loss = valid_loss
				self.best_model = deepcopy(self.local_model)

	def update_global(self, global_model):
		self.local_model = deepcopy(global_model)
		