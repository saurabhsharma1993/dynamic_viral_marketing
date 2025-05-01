import pickle
from scipy.io import loadmat
import os
import sys
import pickle as pkl
from sklearn.model_selection import train_test_split
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.utils import k_hop_subgraph
import pdb

root = './data/raw_data'

class Dataset():
    def __init__(self, dataset, small_scale = False):
        a = h5py.File('{}/{}/training_test_dataset.mat'.format(root,dataset), 'r')
        self.dataset = dataset
        self.small_scale = small_scale
        self.reverse_sort = reverse_sort
        user_feats, user_adj = np.array(a['M'],dtype=np.float32), np.array(a['W_users'],dtype=np.float32)
        user_feats = (user_feats !=0).astype(np.float32)    

        user_feats, labels = self.feature_label_split(user_feats)

        if self.small_scale:
            # pdb.set_trace()
            sel_nodes = self.small_scale_split(user_adj, labels)
            user_adj = user_adj[sel_nodes][:,sel_nodes]
            user_feats = user_feats[sel_nodes]
            labels = labels[sel_nodes]

        lcc = self.largest_connected_components(user_adj)
        user_adj = user_adj[lcc][:, lcc]
        user_feats = user_feats[lcc]
        labels = labels[lcc]

        self.adj        = user_adj
        self.features   = user_feats
        self.labels     = labels

        self.idx_train, self.idx_val, self.idx_test = self.get_splits()

    def largest_connected_components(self, adj, n_components=1):
        adj = sp.csr_matrix(adj)
        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    # split the item ratings into user features and labels 
    def feature_label_split(self, user_feats, reverse_sort = True):
        counts_ = (user_feats != 0).sum(0)
        counts_argsort = np.argsort(counts_)
        if not reverse_sort:
            label_idx = counts_argsort[0]
        else:
            label_idx = counts_argsort[10]
        labels = user_feats[:,label_idx]
        features = np.concatenate((user_feats[:,:label_idx],user_feats[:,label_idx+1:]),1)
        return features, labels

    def get_splits(self):
        size = len(self.labels)
        num_train, num_val = int(0.1*size), int(0.1*size)
        num_test = size - num_train - num_val
        adopters = list(np.where(self.labels == 1)[0])
        non_adopters = list(np.where(self.labels == 0)[0])
        
        idx_train = np.arange(size)
        idx_val = idx_train
        idx_test = idx_train
        
        return idx_train, idx_val, idx_test

    def small_scale_split(self, adj, labels, reverse_sort = True):
        degrees = adj.sum(1)
        if not reverse_sort:
            deg_sort = np.argsort(degrees)[::-1]          # keep highest degree nodes
        else:
            deg_sort = np.argsort(degrees)                  # keep lowest degree nodes
        sel_nodes = list(np.where(labels == 1)[0])
        for node in deg_sort:
            if node not in sel_nodes:
                sel_nodes.append(node)
            if reverse_sort:
                if self.dataset == 'flixster' and len(sel_nodes) == 2750:   # flixster
                    break
                if self.dataset == 'ciao' and len(sel_nodes) == 4500:       # ciao
                    break
                if self.dataset == 'epinions' and len(sel_nodes) == 14650:  # epinions
                    break
            else:
                if len(sel_nodes) == 1000:
                    break
        return sel_nodes
