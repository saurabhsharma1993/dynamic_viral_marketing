## Load data and train a GCN model

import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
import pickle
import math
import os
import pdb
from pprint import pprint
import sys
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from dataset import Dataset
from models import *
from utils import *

def compute_unaffected_set(victim_set, modified_adj, modified_features):
    
    orig_output = model.predict(features, adj)
    new_output = model.predict(modified_features, modified_adj)
    unaffected_set = np.array(victim_set)[(orig_output[victim_set,:] == new_output[victim_set,:]).all(1).data.cpu().numpy()]
    
    return unaffected_set

def select_nodes(target_gcn=None,community='attack'):

    with torch.no_grad():
        output = target_gcn.predict(features,adj)
        pred_labels = output.argmax(1)

        if community == 'attack':
            nodes = []
            # for idx in (idx_train + idx_val):
            for idx in idx_train:   # idx_train = idx_val = idx_test
                if pred_labels[idx] == 1: # nodes classified into attack community
                    nodes.append(idx)
            return nodes
        else:
            nodes = []
            for idx in idx_test:
                pred = output[idx].argmax()
                if pred == 0:     # keep all nodes classified as non-adopters
                    nodes.append(idx) 
            return nodes

def check_adoption(adj, features, target_node, source_nodes, gcn=None):
    with torch.no_grad():
        output = gcn.predict(features, adj)
        probs = torch.softmax(output[target_node],dim=0)
        margin = classification_margin(output[target_node],labels[target_node])
        preds = output.argmax(1)
        pred = preds[target_node]
        backflips = (preds[source_nodes] == 0).sum() # adopters who went back to non-adopters. a continuous model of adoption.
        otherflips = (preds[target_nodes] == 1).sum() - pred # other non-adopters who flipped
        return pred, margin, backflips, otherflips
  
def target_influence_lookahead(target_node, target_set, modified_adj, modified_features, switch_k = 2):
    
    out = model.predict(modified_features,modified_adj)
    
    # obj is sum LM between attack and target classes.
    obj = torch.sum(out[target_set,1] - out[target_set,0]) 

    # get gradient on target attributes
    grad_feats = torch.autograd.grad(obj, modified_features)[0][target_node,:]                                          
    grad_feats = grad_feats * (-2*modified_features[target_node,:] + 1)    # considering discret flips
    grad_feats_sort, sort_inds = torch.sort(grad_feats)
    
    # filter nnp gradient
    nnp_inds = grad_feats_sort <= 0
    grad_feats_sort, sort_inds = grad_feats_sort[~nnp_inds], sort_inds[~nnp_inds]  
    
    # make top-k attr perts of target node
    grad_feats_argmax = sort_inds[-switch_k:]
    value = -2 * modified_features[target_node, grad_feats_argmax] + 1       
    pert_features = modified_features.detach().clone()  # weird memory leak
    pert_features[target_node, grad_feats_argmax] += value
    
    # now gradient of 2nd order perts on edges
    out_lookahead = model.predict(pert_features,modified_adj)

    obj = torch.sum(out_lookahead[target_set,1] - out_lookahead[target_set,0]) 
    
    grad_adj = torch.autograd.grad(obj, modified_adj)[0]
    grad_adj[target_node,target_node] = -10                          # no self loops
    grad_adj = (grad_adj + grad_adj.T)[target_node,:]
    grad_adj = grad_adj * (-modified_adj[target_node,:] + 1)         # considering insertions only
    grad_adj = grad_adj[target_set]
    
    # filter nnp gradient
    nnp_inds = grad_adj <= 0
    grad_adj = grad_adj[~nnp_inds]
    
    # sum all 2nd order nnn gradients
    influence = torch.sum(grad_adj).item() / len(target_set)
        
    return influence, grad_feats_argmax, pert_features

def local_attack_step(target_nodes,source_nodes,switch_k,pert_adj,pert_feats,demand=None):
    
    def score(target_nodes, source_nodes, switch_k, pert_adj, pert_feats):

        best_budget, best_influence, best_target_perts = math.inf, 0, None
        best_target_node, best_grad_argmax, best_grad_feats_argmax = None, None, None
        best_modified_adj, best_modified_features, best_pert_features = None, None, None

        num_nodes = adj.shape[0]
        num_feats = features.shape[1]
        
        container_adj = adj.detach().clone()                # containers for gradients
        container_adj.requires_grad = True
        
        container_features = features.detach().clone()      # containers for gradients
        container_features.requires_grad = True
        
        out = model.predict(container_features,container_adj)
        degs_target_nodes = degrees[np.array(target_nodes)]
        degs_sort = np.argsort(degs_target_nodes.cpu().numpy())
        target_nodes = list(np.array(target_nodes)[degs_sort])  # sort by deg to get easier budgets first
        
        for ind, target_node in tqdm(enumerate(target_nodes)):     
                
            # pdb.set_trace()
            loss = F.cross_entropy(out[[target_node]], torch.Tensor([0]).long().to(device))    # adoption loss
            grad_adj, grad_feats = torch.autograd.grad(loss, [container_adj, container_features], retain_graph = True)
            # grad_adj = torch.zeros_like(adj)
            # grad_feats = torch.zeros_like(features)
            # pert_inds_comb = torch.tensor([[0,0],[1,1]])
            pert_inds_comb = None
            # pdb.set_trace()

            def check_budget(budget=1,pert_inds_comb=None,grad_adj=None,grad_feats=None,debug=False):
                n_perturbations = int(budget)
                if pert_inds_comb is None:    
                    with torch.no_grad():
                        # bidirection
                        # pdb.set_trace()
                        grad_adj = (grad_adj + grad_adj.T)/2 * (-container_adj + 1)            # deletions get zeroed out
                        # grad_feats = grad_feats * (-container_features + 1)                    # only turn on features
                        # restrict set of perturbations   
                        grad_adj[np.arange(num_nodes),np.arange(num_nodes)] = 0               # self loops disallowed            
                        grad_adj[np.repeat(target_nodes,len(target_nodes)),np.tile(target_nodes,len(target_nodes))] = 0           # zero targets-targets subgraph
                        grad_adj[np.repeat(source_nodes,len(source_nodes)),np.tile(source_nodes,len(source_nodes))] = 0           # zero sources-sources subgraph
                        # don't modify prev flips
                        if prev_eflips.shape[1] > 0:
                            grad_adj[np.array(prev_eflips)[0,:],np.array(prev_eflips)[1,:]] = 0 
                            grad_adj[np.array(prev_eflips)[1,:],np.array(prev_eflips)[0,:]] = 0 
                        if prev_fflips.shape[1] > 0:
                            grad_feats[np.array(prev_fflips)[0,:],np.array(prev_fflips)[1,:]] = 0 
                        grad_feats[target_nodes,:] = 0                                        # zero targets-features submat

                        comb_grad = torch.cat((grad_adj,grad_feats),dim=1)
                        comb_grad = comb_grad.flatten()
                        nnn_idx = torch.where(comb_grad > 0)[0]
                        nnn_vals = comb_grad[nnn_idx]
                        
                        # pdb.set_trace()
                        grad_sort, aux_inds = torch.sort(nnn_vals)
                        grad_sort_comb, pert_inds_comb = nnn_vals[aux_inds], nnn_idx[aux_inds]
                        pert_inds_comb = torch.stack([pert_inds_comb//(num_nodes+num_feats), pert_inds_comb%(num_nodes+num_feats)]).long()
                        
                        # only check lower triangular indices of adj
                        select = torch.logical_or(pert_inds_comb[0,:] < pert_inds_comb[1,:], pert_inds_comb[1,:] >= num_nodes)       
                        pert_inds_comb = pert_inds_comb[:,select]
                        
                def pick_perts():
                    if pert_adj and pert_feats:
                        comb_grad_argmax = pert_inds_comb[:,-n_perturbations:]
                        grad_argmax_adj, grad_argmax_feats = comb_grad_argmax[:,comb_grad_argmax[1,:] < num_nodes], \
                                                             comb_grad_argmax[:,comb_grad_argmax[1,:] >= num_nodes]
                        grad_argmax_feats[1,:] = grad_argmax_feats[1,:] - num_nodes 
                    elif pert_adj and not pert_feats:
                        pert_inds_adj = pert_inds_comb[:,pert_inds_comb[1,:] < num_nodes]
                        grad_argmax_adj, grad_argmax_feats = pert_inds_adj[:,-n_perturbations:], torch.empty([2,0])
                    elif not pert_adj and pert_feats:
                        pert_inds_feats = pert_inds_comb[:,pert_inds_comb[1,:] >= num_nodes]
                        grad_argmax_adj, grad_argmax_feats = torch.empty([2,0]), pert_inds_feats[:,-n_perturbations:]
                        grad_argmax_feats[1,:] = grad_argmax_feats[1,:] - num_nodes 
                    else:
                        raise Exception('Must pert at least one of edges and feats.')
                    return grad_argmax_adj, grad_argmax_feats

                grad_argmax, grad_feats_argmax = pick_perts()
                        
                modified_adj, modified_features = adj.detach().clone(), features.detach().clone()
                modified_adj.requires_grad, modified_features.requires_grad = True, True

                # make updates
                if pert_adj:
                    value = -modified_adj[grad_argmax[0],grad_argmax[1]] + 1       
                    modified_adj.data[grad_argmax[0],grad_argmax[1]] += value
                    modified_adj.data[grad_argmax[1],grad_argmax[0]] += value
                
                if pert_feats:
                    value = -2*modified_features[grad_feats_argmax[0],grad_feats_argmax[1]] + 1              # forgot to do this earlier. would increase budget a bit and turn off features that it thought it could increase its weight.   
                    # value = -modified_features[grad_feats_argmax[0],grad_feats_argmax[1]] + 1                # only turn on features
                    modified_features.data[grad_feats_argmax[0],grad_feats_argmax[1]] += value
                
                pred, margin, backflips, otherflips = check_adoption(modified_adj, modified_features, target_node, source_nodes, gcn=model)
                
                if not debug:
                    return pred, margin, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips, otherflips
                else:
                    return pred, margin, grad_argmax, grad_feats_argmax, pert_inds_comb, grad_sort_comb

            def bin_search(target_node, source_nodes, pert_inds_comb, adj, features, model):

                l_budget, u_budget = 0, degrees[target_node].item()
                l_pred, _, backflips, otherflips = check_adoption(adj, features, target_node, source_nodes, gcn=model)
                if l_pred == 1:
                    budget = l_budget + backflips - otherflips
                    act_budget = l_budget
                    modified_adj, modified_features = adj.detach().clone(), features.detach().clone()
                    modified_adj.requires_grad, modified_features.requires_grad = True, True        
                    grad_argmax, grad_feats_argmax = None, None

                else:
                    u_pred = 0
                    max_budget = torch.max(degrees)   # for reasons of computational efficiency the search can't be unbounded. the max degree is consistent with observed network statistics.
                    
                    while u_pred == 0:
                        budget = l_budget             # conservative estimate. don't care about other/back flips.
                        if u_budget > max_budget.item() or budget > best_budget:
                            u_budget, l_budget, budget = math.inf, math.inf, math.inf
                            act_budget = u_budget
                            modified_adj, modified_features = adj.detach().clone(), features.detach().clone()
                            modified_adj.requires_grad, modified_features.requires_grad = True, True        
                            grad_argmax, grad_feats_argmax = None, None    
                            break
                        u_pred, _, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips, otherflips = check_budget(u_budget,pert_inds_comb,grad_adj,grad_feats)
                        if u_pred == 0:
                            l_budget = u_budget
                            u_budget = 2*u_budget
                        
                    while u_budget - l_budget > 1:    
                        budget = l_budget            # conservative estimate. don't care about other/back flips.
                        if budget > best_budget:
                            u_budget, l_budget, budget = math.inf, math.inf, math.inf
                            act_budget = u_budget
                            modified_adj, modified_features = adj.detach().clone(), features.detach().clone()
                            modified_adj.requires_grad, modified_features.requires_grad = True, True        
                            grad_argmax, grad_feats_argmax = None, None    
                            break
                        c_budget = (l_budget + u_budget) // 2
                        c_pred, _, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips, otherflips = check_budget(c_budget,pert_inds_comb,grad_adj,grad_feats)
                        if c_pred == 0:
                            l_budget = c_budget
                        else:
                            u_budget = c_budget
                    
                    if u_budget != math.inf:
                        u_pred, _, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips, otherflips = check_budget(u_budget,pert_inds_comb,grad_adj,grad_feats)
                        assert u_pred == 1, 'u_pred is not equal to 1'
                        act_budget = u_budget
                        budget = u_budget + backflips - otherflips # account for backflips and otherflips

                return budget, act_budget, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips, otherflips

            if target_node in budget_dict and target_node in unaffected_set:
                budget, actual_budget = budget_dict[target_node] # [total_cost,actual_budget]
            else:
                budget, act_budget, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips, otherflips = bin_search(target_node, source_nodes, pert_inds_comb, adj, features, model)
                budget_dict[target_node] = [budget,act_budget]
                
            target_set = deepcopy(target_nodes)
            target_set.remove(target_node) 
            
            if args.infls_pert: 
                if target_node in influence_dict and target_node in unaffected_set:
                    influence = influence_dict[target_node]
                else:
                    influence, target_perts, pert_features = target_influence_lookahead(target_node, target_set, modified_adj, modified_features, switch_k)
                    influence_dict[target_node] = influence
            else:
                influence, target_perts, pert_features = 0, None, None

            if budget < best_budget or (budget == best_budget and influence >= best_influence):
                
                if target_node in unaffected_set:
                    budget, act_budget, grad_argmax, grad_feats_argmax, pert_inds_comb, modified_adj, modified_features, backflips, otherflips = bin_search(target_node, source_nodes, pert_inds_comb, adj, features, model)
                    budget_dict[target_node] = [budget,act_budget]
                    if args.infls_pert:
                        influence, target_perts, pert_features = target_influence_lookahead(target_node, target_set, modified_adj, modified_features, switch_k)                    
                        influence_dict[target_node] = influence
                best_budget = budget
                best_target_node, best_grad_argmax, best_grad_feats_argmax = target_node, grad_argmax, grad_feats_argmax
                best_influence, best_target_perts = influence, target_perts
                best_modified_adj = modified_adj
                best_modified_features = pert_features if args.infls_pert and ((influence != 0 and influence >= args.influence_thresh) or args.fixed_ip) else modified_features
                # best_modified_features = modified_features. turn on for ablation.

        # pdb.set_trace()
        return best_target_node, best_budget, best_grad_argmax, best_grad_feats_argmax, best_influence, best_target_perts, best_modified_adj, best_modified_features

    target_node, budget, grad_argmax, grad_feats_argmax, influence, target_perts, modified_adj, modified_features = score(target_nodes, source_nodes, switch_k, pert_adj, pert_feats)
    global unaffected_set
    unaffected_set = compute_unaffected_set(target_nodes,modified_adj,modified_features)
    
    if budget == math.inf or budget == 1e13:
        return target_nodes, source_nodes, None, None, None, None, None, None
    # elif budget > 0:
    else:
        # make updates
        global adj 
        adj = modified_adj.detach().clone()
        global features
        features = modified_features.detach().clone()
    # else: 
    #     pass

    budget = grad_argmax.shape[1] + grad_feats_argmax.shape[1] # actual budget

    all_ips = {}
    if args.infls_pert and ((influence != 0 and influence >= args.influence_thresh) or args.fixed_ip):
        print_write(['Influence is good at node: {}'.format(target_node)],log_file)
        print_write(['Target perts.'],log_file)
        print_write([target_perts],log_file)
        budget = budget + len(target_perts) # add perts at converted target to budget
        all_ips[target_node] = target_perts
    
    print_write(["Target node: {}, Budget: {}".format(target_node,budget)],log_file)
    print_write(["Attack edges."],log_file)
    print_write([grad_argmax],log_file)
    print_write(["Attack feature flips."],log_file)
    print_write([grad_feats_argmax],log_file)

    new_target_nodes = select_nodes(model,'target')
    converted_nodes = list(set.difference(set(target_nodes),set(new_target_nodes)))

    print_write(["Converted nodes."],log_file)
    print_write([converted_nodes],log_file)

    assert target_node in converted_nodes, 'target is not converted'
    assert len(converted_nodes) > 0, 'nothing is converted'
    
    converted_nodes.remove(target_node)
    converted_nodes = [target_node] + converted_nodes # for tracking purpose, keep target at index 0.
    target_nodes = new_target_nodes
    source_nodes = source_nodes + converted_nodes
    
    backflips = list(set.intersection(set(target_nodes),set(source_nodes)))
    for node in backflips:
        print_write(['Node {} backflipped.'.format(node)],log_file)
        source_nodes.remove(node)
    assert len(set.intersection(set(target_nodes),set(source_nodes))) == 0, 'Inconsistent set of targets and attackers'

    return target_nodes, source_nodes, converted_nodes, budget, influence, all_ips, grad_argmax, grad_feats_argmax

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed.')
    parser.add_argument('--dataset', type=str, default='flixster', choices=['flixster', 'epinions', 'ciao'], help='dataset')
    parser.add_argument('--model', type=str, default='sage', choices=['gcn', 'sage'], help='dataset')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--switch_k', type=int, default=2, help='Number of attr perts to make at converted target.')
    parser.add_argument('--perc_atkrs', type=int, default=100, help='Percentage of maxm attackers.')
    parser.add_argument('--influence_thresh', type=float, default=2.7, help='Influence cutoff to make attr perts at converted target.')
    parser.add_argument('--log_dir', type=str, default='', help='Weight of corruption objective.')
    parser.add_argument('--exp', type=str, default='default', help='name of the experiment.')
    parser.add_argument('--no_adj_pert', default=False, action='store_true')
    parser.add_argument('--no_feat_pert', default=False, action='store_true')
    parser.add_argument('--no_infl_pert', default=False, action='store_true')
    parser.add_argument('--fixed_ip', default=False, action='store_true')
    parser.add_argument('--demand', type=int, default=500, help='Number to convert')
    parser.add_argument('--small', default=False, action='store_true', help='Small scale dataset')
    parser.add_argument('--reverse', default=False, action='store_true', help='Small scale dataset')

    args = parser.parse_args()
    args.pert_adj, args.pert_feats = not args.no_adj_pert, not args.no_feat_pert
    args.infls_pert = not args.no_infl_pert or args.fixed_ip
    args.cuda = torch.cuda.is_available()
    if args.small:
        args.exp = 'small_' + args.exp
    if args.reverse:
        args.exp = 'reverse_' + args.exp

    root_dir = './logs'
    log_dir = os.path.join(root_dir,'{}'.format(args.dataset))
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")
    log_file = os.path.join(log_dir,'{}.txt'.format(args.exp))

    for arg in vars(args):
        print_write(['{}: {}'.format(arg, getattr(args, arg))],log_file)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    small_scale = args.small
    reverse = args.reverse
    data = Dataset(args.dataset,small_scale=small_scale,reverse_sort=reverse)
    adj, features, labels = data.adj, data.features, data.labels
    idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

    num_nodes = adj.shape[0]
    num_feats = features.shape[1]
    print_write(["Num of nodes: {}, num of feats: {}".format(num_nodes,num_feats)],log_file)

    features, adj, labels = torch.from_numpy(features).float(), torch.from_numpy(adj).float(), torch.from_numpy(labels).long()
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    degrees = torch.sum(adj,dim=0)

    if args.model == 'gcn':
        model = Trainable_GCN(features.shape[1],64,labels.max().item()+1)
    else:
        model = Trainable_GCN(features.shape[1],64,labels.max().item()+1,use_sage=True)
    model = model.to(device)
    results, train_model = None, True

    if os.path.isfile(os.path.join(log_dir,'{}.pkl'.format(args.exp))):
        with open(os.path.join(log_dir,'{}.pkl'.format(args.exp)), 'rb') as f:
            results = pickle.load(f)  
        if 'state_dict' in results:
            model.load_state_dict(results['state_dict'])
            train_model = False
            print_write(['Loaded pre-trained model.'],log_file)

    if train_model:
        model.fit(device, features, adj, labels, idx_train, idx_val, balanced=True)

    out = model.predict(features,adj)
    pred_labels = out.argmax(dim=1)
    test_acc = (pred_labels[idx_test] == labels[idx_test]).sum() / len(idx_test)
    tst_0, tst_1 = labels[idx_test] == 0, labels[idx_test] == 1
    tst_acc0 = (out[idx_test][tst_0].argmax(dim=1) == labels[idx_test][tst_0]).sum() / sum(tst_0) 
    tst_acc1 = (out[idx_test][tst_1].argmax(dim=1) == labels[idx_test][tst_1]).sum() / sum(tst_1) 
    test_acc = (tst_acc0 + tst_acc1)/2
    print_write(["Balanced Test acc: {:.4f}, Class 0 acc: {:.4f}, Class 1 acc: {:.4f}".format(test_acc.item(), tst_acc0.item(), tst_acc1.item())],log_file)

    attack_nodes = select_nodes(model,'attack') # source and attack is used interchangeably
    target_nodes = select_nodes(model,'target')

    num_nodes = adj.shape[0]
    converted_nodes, budgets, influences, attack_edges, attribute_switches, infl_perts, num_converted = [], [], [], [], [], [], []
    prev_eflips, prev_fflips = np.empty((2,0)), np.empty((2,0))
    init_source_size = len(attack_nodes)
    prev_run, prev_runtime = False, 0
    unaffected_set = np.array([])
    budget_dict, influence_dict = {}, {}

    def init_state():
        if os.path.isfile(os.path.join(log_dir,'{}.pkl'.format(args.exp))):
            with open(os.path.join(log_dir,'{}.pkl'.format(args.exp)), 'rb') as f:
                results = pickle.load(f)  
            global converted_nodes, budgets, influences, attack_edges, attribute_switches, infl_perts, num_converted, prev_eflips, prev_fflips, adj, features, attack_nodes, target_nodes
            converted_nodes = results['converted_nodes']
            budgets = results['budgets']
            influences = results['influences']
            attack_edges = results['attack_edges']
            attribute_switches = results['attribute_switches']
            infl_perts = results['infl_perts']
            num_converted = results['num_converted']
            prev_eflips = np.hstack(attack_edges)
            prev_fflips = np.hstack(attribute_switches)
            if len(infl_perts) > 0:
                prev_fflips_ = np.hstack([np.vstack([np.repeat(k,len(v)),v]) for (k,v) in infl_perts])
                prev_fflips = np.hstack([prev_fflips,prev_fflips_])
            adj = torch.from_numpy(update_adj(data.adj, attack_edges)).float().to(device)
            features = torch.from_numpy(update_features(data.features, attribute_switches, infl_perts)).float().to(device)    
            attack_nodes = select_nodes(model,'attack')
            target_nodes = select_nodes(model,'target')
            global prev_run, prev_runtime
            prev_run, prev_runtime = True, results['runtime']
            print_write(['Loaded attack from {} steps with {} attack nodes ({} converted).'.format(len(budgets),len(attack_nodes),num_converted[-1])],log_file)
    init_state()
    # pdb.set_trace()

    start_time = time()
    last_best, time_since = init_source_size, 0
    demand = args.demand

    while len(attack_nodes) < args.demand:
        sys.stdout.flush()

        target_nodes, attack_nodes, converted_node_list, budget, influence, all_ips, grad_argmax, grad_feats_argmax = local_attack_step(target_nodes, attack_nodes, args.switch_k, args.pert_adj, args.pert_feats, demand)
        if converted_node_list == None:
            print_write(['Attack failed. No target node is flippable.'],log_file)
            break
        # converted_nodes = converted_nodes + converted_node_list
        converted_nodes.append(converted_node_list) 
        budgets.append(budget)                                                                 
        influences.append(influence)
        attack_edges.append(to_numpy(grad_argmax))
        attribute_switches.append(to_numpy(grad_feats_argmax))
        infl_perts.extend([(k,to_numpy(v)) for (k,v) in all_ips.items()])
        num_converted.append(len(attack_nodes) - init_source_size)
        
        prev_eflips = np.hstack(attack_edges)
        prev_fflips = np.hstack(attribute_switches)
        if len(infl_perts) > 0:
            prev_fflips_ = np.hstack([np.vstack([np.repeat(k,len(v)),v]) for (k,v) in infl_perts])
            prev_fflips = np.hstack([prev_fflips,prev_fflips_])

        # stopping criterion. if target set size doesn't improve for 20 steps, stop.
        if len(attack_nodes) > last_best:
            last_best, time_since = len(attack_nodes), 0
        else:
            time_since += 1
        if time_since >= 40:
            print_write(['Failure. Attack didn\'t decrease target size for 40 steps.'],log_file)
            break

        end_time = time()
        runtime = end_time - start_time
        if prev_run:
            runtime = runtime + prev_runtime
        print_write(["Time taken for {} steps: {:.2f} secs. Estimated total: {:.2f} secs.".format(len(budgets),runtime,runtime*(args.demand)/len(budgets))],log_file)
        
        total_budget = sum(budgets)
    
        results = {'args': args,\
                    'converted_nodes': converted_nodes,\
                    'budgets': budgets,\
                    'total_budget': total_budget,\
                    'influences': influences,\
                    'attack_edges': attack_edges,\
                    'runtime': runtime,\
                    'attribute_switches': attribute_switches,\
                    'infl_perts': infl_perts,\
                    'num_converted': num_converted,\
                    'state_dict': model.state_dict()}

        with open(os.path.join(log_dir,'{}.pkl'.format(args.exp)), 'wb') as f:
            pickle.dump(results,f)

        if len(attack_nodes) >= args.demand: # new stopping criterion
            break

    end_time = time()
    runtime, total_budget = end_time - start_time, sum(budgets)
    if prev_run:
        runtime = runtime + prev_runtime
    
    print_write(["Time taken to attack: {:.2f} seconds".format(runtime)],log_file)
    print_write(["Total budget: {}".format(total_budget)],log_file)
