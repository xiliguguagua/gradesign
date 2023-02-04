import torch
import torch.nn as nn
from copy import deepcopy

from torch_network import emnist_net, cifar10_net

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class User(object):
	"""docstring for User"""
	def __init__(self, uid):
		super(User, self).__init__()
		self.id = uid
		self.lr = lr
		self.ismalice = is_m
		self.max_iteration = max_it
		self.min_iteration = min_it

		self.local_dataloder = ld
		self.CELoss = nn.CrossEntropyLoss().to(device)
		if task == 'emnist':
			self.local_model = emnist_net().to(device)
		else:
			self.local_model = cifar10_net().to(device)


	def local_train(self):
		optm = torch.optim.SGD(self.local_model.parameters(), lr=self.lr)
		best_loss = 1e10

		for _ in range(self.max_iteration):

			self.local_model.train()
			for data in self.local_dataloder['train']:
				train_data, train_label = data
				train_data = train_data.to(device)
				train_label = train_label.to(device)

				output = self.local_model(train_data)
				optm.zero_grad()
				ce_loss = self.CELoss(output, train_label)
				ce_loss.backward()
				optm.step()

			self.local_model.eval()
			valid_loss = 0
			with torch.no_grad():
				for data in self.local_dataloder['valid']:
					valid_data, valid_label = data
					valid_data = valid_data.to(device)
					valid_label = valid_label.to(device)

					output = self.local_model(valid_data)
					valid_loss += self.CELoss(output, valid_label)

			if valid_loss > best_loss and _ > self.min_iteration:  # early stop when overfitting
				self.local_model = self.best_model
				return
			if valid_loss < best_loss or best_loss == 1e10:
				best_loss = valid_loss
				self.best_model = deepcopy(self.local_model)

		self.local_model = self.best_model

	def clipping_perturbation(self):
		parameters = []
		for p in self.local_model.parameters():
			parameters.append(p.data)

	def update_global(self, global_model):
		self.local_model = deepcopy(global_model)
		