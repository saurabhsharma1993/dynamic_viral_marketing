import pdb
import torch
import torch.nn as nn
import torch_geometric as pyg
from utils import *
from copy import deepcopy

class GCNConv(nn.Module):
	def __init__(self, inc, outc):
		super().__init__()
		self.inc, self.outc = inc, outc
		self.feat_transform = nn.Linear(inc,outc)
		self.feat_transform.reset_parameters()
	def forward(self, X, prop_matrix):
		out = prop_matrix @ self.feat_transform(X) 					
		return out

# expects the prop matrix, not the adjacency like pyg model
class Trainable_GCN(nn.Module):
	def __init__(self, inc, hidden, outc, dropout = 0.5, use_sage=False):
		super().__init__()
		self.conv1 = GCNConv(inc,hidden)
		self.conv2 = GCNConv(hidden,outc)
		self.activation = nn.ReLU()
		# self.dropout = nn.Dropout(dropout)
		self.use_sage = use_sage
	def forward(self,X,prop_matrix):
		# out = self.dropout(self.activation(self.conv1(X,prop_matrix)))
		out = self.activation(self.conv1(X,prop_matrix))
		out = self.conv2(out,prop_matrix)
		return out
	def fit(self, device, features, adj, labels, idx_train, idx_val, balanced=False):
		if self.use_sage:
			print('Training Graph Sage.')
		else:
			print('Training GCN.')
		
		if self.use_sage:
			prop = normalize_adj_tensor_sage(adj)
		else:
			prop = normalize_adj_tensor(adj)
		epochs = 100 				
		lr, wd = 1e-2, 5e-4

		optimizer = torch.optim.Adam(self.parameters(),lr=lr, weight_decay=wd)
		scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

		if balanced:
			self.balanced = True
			num_class0, num_class1 = (labels[idx_train] == 0).sum(), (labels[idx_train] == 1).sum()
			bal_weights = torch.Tensor([num_class1/num_class0,1])
			criterion = nn.CrossEntropyLoss(weight=bal_weights).to(device)
		else:
			self.balanced = False
			criterion = nn.CrossEntropyLoss().to(device)

		best_acc, best_epoch = 0, None
		self.train()
		self.epoch_accs = []

		for epoch in range(1, epochs+1):
			optimizer.zero_grad()
			out = self.forward(features,prop)
			loss = criterion(out[idx_train],labels[idx_train])
			loss.backward()
			optimizer.step()

			out = self.forward(features,prop)
			trn_0, trn_1 = labels[idx_train] == 0, labels[idx_train] == 1
			trn_acc0 = (out[idx_train][trn_0].argmax(dim=1) == labels[idx_train][trn_0]).sum() / sum(trn_0) 
			trn_acc1 = (out[idx_train][trn_1].argmax(dim=1) == labels[idx_train][trn_1]).sum() / sum(trn_1) 
			train_acc = (trn_acc0 + trn_acc1)/2

			self.epoch_accs.append(train_acc)
			if epoch % 20 == 0:
				print('Loss: {:.2f}, Train_acc: {:.2f}, Val_acc: {:.2f}'.format(loss.item(),train_acc,train_acc))

			if train_acc >= best_acc:
				best_acc = train_acc
				best_epoch = epoch
				best_state = deepcopy(self.state_dict())
			elif epoch >= best_epoch + 20:
				break

		print(best_acc, best_epoch)
		self.load_state_dict(best_state) # reload best model
	def predict(self,features,adj):
		self.eval()
		if self.use_sage:
			prop = normalize_adj_tensor_sage(adj)
		else:
			prop = normalize_adj_tensor(adj)
		out = self.forward(features,prop)
		return out