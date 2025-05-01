from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import torch_geometric as pyg
import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import scipy

def print_write(print_str, log_file):
    print(*print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        print(*print_str, file=f)

def to_numpy(rand_cuda_var):
    if rand_cuda_var == None:
        return rand_cuda_var
    else:
        return rand_cuda_var.data.cpu().numpy()

def update_adj(adj, attack_edges):
	adj = deepcopy(adj)
	for edge_attacks in attack_edges:
		if edge_attacks.shape[1] > 0:
			srcs, dests = edge_attacks[0,:], edge_attacks[1,:]
			vals = -adj[srcs,dests] + 1
			adj[srcs,dests] += vals
			adj[dests,srcs] += vals
			# print('added')
	return adj

def update_features(features, attribute_switches, infl_perts):
	features = deepcopy(features)
	for attr_switch in attribute_switches:
		if attr_switch.shape[1] > 0:
			nodes, switches = attr_switch[0,:], attr_switch[1,:]
			vals = -2*features[nodes, switches] + 1
			features[nodes,switches] += vals
	for infl_pert in infl_perts:
		node, perts = infl_pert
		vals = -2*features[node, perts] + 1
		features[node,perts] += vals
	return features

def normalize_adj_tensor(adj):
	device = adj.device
	mx = adj + torch.eye(adj.shape[0]).to(device)
	rowsum = mx.sum(1)
	r_inv = rowsum.pow(-1/2).flatten()
	r_inv[torch.isinf(r_inv)] = 0.
	r_mat_inv = torch.diag(r_inv)
	mx = r_mat_inv @ mx
	mx = mx @ r_mat_inv
	return mx.to(device)

def normalize_adj_tensor_sage(adj):
	device = adj.device
	mx = adj + torch.eye(adj.shape[0]).to(device)
	rowsum = mx.sum(1)
	r_inv = rowsum.pow(-1).flatten()
	r_inv[torch.isinf(r_inv)] = 0.
	r_mat_inv = torch.diag(r_inv)
	mx = r_mat_inv @ mx
	return mx.to(device)

def classification_margin(output, true_label):

	probs = torch.softmax(output,dim=0)
	probs_true_label = probs[true_label].clone()
	probs[true_label] = 0
	probs_best_second_class = probs[probs.argmax()]
	return (probs_true_label - probs_best_second_class).item()

def two_hop_nbrs(node, adj):
	nbrs = [n.item() for n in torch.where(adj[node,:])[0]]
	two_hop_nbrs = []
	two_hop_nbrs.extend(nbrs)
	for nbr in nbrs:
		two_hops = [n.item() for n in torch.where(adj[nbr,:])[0]]
		two_hop_nbrs.extend(two_hops)
	two_hop_nbrs = np.unique(np.array(two_hop_nbrs))
	return two_hop_nbrs

