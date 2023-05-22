from operator import index
import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, dense_to_sparse, to_dense_adj, is_undirected
from torch_geometric.transforms import NormalizeFeatures
from deeprobust.graph.data import Dataset, PrePtbDataset
import json
import scipy.sparse as sp
from deeprobust.graph.global_attack import Random
from deeprobust.graph import utils

def get_data(root, name, attack, ptb_rate):
    if name in ['cora', 'citeseer', 'pubmed', 'polblogs']:
        data = Dataset(root=root, name=name, setting='prognn')
        adj, features, labels = data.adj, data.features, data.labels 
        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test

        dataset = Data()
        dataset.x = torch.from_numpy(features.toarray()).float()
        dataset.y = torch.from_numpy(labels).long()
        dataset.edge_index = dense_to_sparse(torch.from_numpy(adj.toarray()))[0].long()
        dataset.edge_index = to_undirected(dataset.edge_index)
        
        dataset.train_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_train)).bool()
        dataset.val_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_val)).bool()
        dataset.test_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_test)).bool()

    elif name in ['photo', 'computers']:
        from torch_geometric.datasets import Amazon
        dataset = Amazon(root, name)[0]
        dataset.edge_index = to_undirected(dataset.edge_index)
        adj = to_dense_adj(dataset.edge_index, max_num_nodes=dataset.x.size(0))[0]
        adj = utils.to_scipy(adj)
            
        if os.path.isfile(f'{root}{name}_train_mask.pt'):
            dataset.train_mask = torch.load(f'{root}{name}_train_mask.pt')
            dataset.val_mask = torch.load(f'{root}{name}_val_mask.pt')
            dataset.test_mask = torch.load(f'{root}{name}_test_mask.pt')
        else:
            print('Creating Mask')
            dataset = create_masks(dataset)
            torch.save(dataset.train_mask, f'{root}{name}_train_mask.pt')
            torch.save(dataset.val_mask, f'{root}{name}_val_mask.pt')
            torch.save(dataset.test_mask, f'{root}{name}_test_mask.pt')

    elif name in ['Automotive_0', 'Automotive_100', 'Pet_Supplies_0', 'Pet_Supplies_200', 'Patio_Lawn_and_Garden_0', 'Patio_Lawn_and_Garden_100']:
        dataset = torch.load(f'{root}{name}.pt')        
        return [dataset]

    if attack == 'clean':
        return [dataset]
        
    if attack == 'clean_net':
        with open(f'{root}{name}_nettacked_nodes.json') as json_file:
            ptb_idx = json.load(json_file)
        idx_test_att = ptb_idx['attacked_test_nodes']
        dataset.test_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_test_att)).bool()
        return [dataset]

    elif attack == 'noisy_str_meta':
        if name in ['photo', 'computers']:
            perturbed_adj = sp.load_npz(f'{root}{name}_meta_adj_0.25.npz')
            dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
            dataset.edge_index = to_undirected(dataset.edge_index)
        else:
            perturbed_data = PrePtbDataset(root=root, name=name, attack_method='meta', ptb_rate=0.25)
            perturbed_adj = perturbed_data.adj
            dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
            dataset.edge_index = to_undirected(dataset.edge_index)
        return [dataset]
    
    elif attack == 'noisy_str_net':    
        perturbed_adj = sp.load_npz(f'{root}{name}_nettack_adj_{5}.0.npz')
        with open(f'{root}{name}_nettacked_nodes.json') as json_file:
            ptb_idx = json.load(json_file)
        
        dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
        dataset.edge_index = to_undirected(dataset.edge_index)
        idx_test_att = ptb_idx['attacked_test_nodes']
        dataset.test_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_test_att)).bool()
        return [dataset]

    elif attack == 'noisy_str_dice_25':    
        perturbed_adj = sp.load_npz(f'{root}{name}_dice_adj_0.25.npz')
        dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
        dataset.edge_index = to_undirected(dataset.edge_index)
        return [dataset]
    
    elif attack == 'noisy_str_rand_25':
        attacker = Random()
        n_perturbations = int(0.25 * (dataset.edge_index.shape[1]//2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj
        dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()            
        dataset.edge_index = to_undirected(dataset.edge_index)
        return [dataset]

    elif attack == 'real_world':
        if name in ['photo', 'computers']:
            perturbed_adj = sp.load_npz(f'{root}{name}_meta_adj_0.25.npz')
            dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
            dataset.edge_index = to_undirected(dataset.edge_index)
        else:
            perturbed_data = PrePtbDataset(root=root, name=name, attack_method='meta', ptb_rate=0.25)
            perturbed_adj = perturbed_data.adj
            dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
            dataset.edge_index = to_undirected(dataset.edge_index)
        
        if name=='polblogs':
            return [dataset]
        elif os.path.isfile(f'{root}{name}_{attack}_x.pt'):
            dataset.x = torch.load(f'{root}{name}_{attack}_x.pt')
        else:
            str_attacked_node_mask = (perturbed_adj > adj).toarray().sum(1).astype(bool)
            str_attacked_node = str_attacked_node_mask.nonzero()[0]
            str_non_attacked_node = (~str_attacked_node_mask).nonzero()[0]
            
            feat_attacked_node1 = np.random.choice(str_attacked_node, len(str_attacked_node)//2, replace=False)
            feat_attacked_node2 = np.random.choice(str_non_attacked_node, len(str_non_attacked_node)//2, replace=False)
            feat_attacked_nodes = np.sort(np.concatenate((feat_attacked_node1, feat_attacked_node2)))
            feat_attacked_mask = np.in1d(np.arange(dataset.x.size(0)), feat_attacked_nodes)
            perturb = torch.randn_like(dataset.x)
            dataset.x[feat_attacked_mask] = dataset.x[feat_attacked_mask] + perturb[feat_attacked_mask]*0.5
            torch.save(dataset.x, f'{root}{name}_{attack}_x.pt')
            print(f'# Only Str-Attacked Nodes: {(str_attacked_node_mask * ~feat_attacked_mask).sum()}, # Only Ft-Attacked Nodes: {(~str_attacked_node_mask * feat_attacked_mask).sum()}, # Both-Attacked Nodes: {(feat_attacked_mask*str_attacked_node_mask).sum()}, # Pure Nodes {(~feat_attacked_mask*~str_attacked_node_mask).sum()}')
        return [dataset]
    
    elif attack == 'real_world_net':
        perturbed_adj = sp.load_npz(f'{root}{name}_nettack_adj_{5}.0.npz')
        with open(f'{root}{name}_nettacked_nodes.json') as json_file:
            ptb_idx = json.load(json_file)
        
        dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
        dataset.edge_index = to_undirected(dataset.edge_index)
        idx_test_att = ptb_idx['attacked_test_nodes']
        dataset.test_mask = torch.from_numpy(np.in1d(np.arange(len(labels)), idx_test_att)).bool()
        
        if name=='polblogs':
            return [dataset]
        elif os.path.isfile(f'{root}{name}_{attack}_x.pt'):
            dataset.x = torch.load(f'{root}{name}_{attack}_x.pt')
        else:
            str_attacked_node_mask = (perturbed_adj > adj).toarray().sum(1).astype(bool)
            str_attacked_node = str_attacked_node_mask.nonzero()[0]
            str_non_attacked_node = (~str_attacked_node_mask).nonzero()[0]
            
            feat_attacked_node1 = np.random.choice(str_attacked_node, len(str_attacked_node)//2, replace=False)
            feat_attacked_node2 = np.random.choice(str_non_attacked_node, len(str_non_attacked_node)//2, replace=False)
            feat_attacked_nodes = np.sort(np.concatenate((feat_attacked_node1, feat_attacked_node2)))
            feat_attacked_mask = np.in1d(np.arange(dataset.x.size(0)), feat_attacked_nodes)
            perturb = torch.randn_like(dataset.x)
            dataset.x[feat_attacked_mask] = dataset.x[feat_attacked_mask] + perturb[feat_attacked_mask]*0.5
            torch.save(dataset.x, f'{root}{name}_{attack}_x.pt')
            print(f'# Only Str-Attacked Nodes: {(str_attacked_node_mask * ~feat_attacked_mask).sum()}, # Only Ft-Attacked Nodes: {(~str_attacked_node_mask * feat_attacked_mask).sum()}, # Both-Attacked Nodes: {(feat_attacked_mask*str_attacked_node_mask).sum()}, # Pure Nodes {(~feat_attacked_mask*~str_attacked_node_mask).sum()}')
        return [dataset]
    
    elif attack == 'real_world_dice_25':
        perturbed_adj = sp.load_npz(f'{root}{name}_dice_adj_0.25.npz')
        dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()
        dataset.edge_index = to_undirected(dataset.edge_index)

        if name=='polblogs':
            return [dataset]
        elif os.path.isfile(f'{root}{name}_{attack}_x.pt'):
            dataset.x = torch.load(f'{root}{name}_{attack}_x.pt')
        else:
            str_attacked_node_mask = (perturbed_adj > adj).toarray().sum(1).astype(bool)
            str_attacked_node = str_attacked_node_mask.nonzero()[0]
            str_non_attacked_node = (~str_attacked_node_mask).nonzero()[0]
            
            feat_attacked_node1 = np.random.choice(str_attacked_node, len(str_attacked_node)//2, replace=False)
            feat_attacked_node2 = np.random.choice(str_non_attacked_node, len(str_non_attacked_node)//2, replace=False)
            feat_attacked_nodes = np.sort(np.concatenate((feat_attacked_node1, feat_attacked_node2)))
            feat_attacked_mask = np.in1d(np.arange(dataset.x.size(0)), feat_attacked_nodes)
            perturb = torch.randn_like(dataset.x)
            dataset.x[feat_attacked_mask] = dataset.x[feat_attacked_mask] + perturb[feat_attacked_mask]*0.5
            torch.save(dataset.x, f'{root}{name}_{attack}_x.pt')
            print(f'# Only Str-Attacked Nodes: {(str_attacked_node_mask * ~feat_attacked_mask).sum()}, # Only Ft-Attacked Nodes: {(~str_attacked_node_mask * feat_attacked_mask).sum()}, # Both-Attacked Nodes: {(feat_attacked_mask*str_attacked_node_mask).sum()}, # Pure Nodes {(~feat_attacked_mask*~str_attacked_node_mask).sum()}')
        return [dataset]
    
    elif attack == 'real_world_rand_25':
        attacker = Random()
        n_perturbations = int(0.25 * (dataset.edge_index.shape[1]//2))
        attacker.attack(adj, n_perturbations, type='add')
        perturbed_adj = attacker.modified_adj
        dataset.edge_index = dense_to_sparse(torch.from_numpy(perturbed_adj.toarray()))[0].long()            
        dataset.edge_index = to_undirected(dataset.edge_index)
        
        if name=='polblogs':
            return [dataset]
        elif os.path.isfile(f'{root}{name}_{attack}_x.pt'):
            dataset.x = torch.load(f'{root}{name}_{attack}_x.pt')
        else:
            str_attacked_node_mask = (perturbed_adj > adj).toarray().sum(1).astype(bool)
            str_attacked_node = str_attacked_node_mask.nonzero()[0]
            str_non_attacked_node = (~str_attacked_node_mask).nonzero()[0]
            
            feat_attacked_node1 = np.random.choice(str_attacked_node, len(str_attacked_node)//2, replace=False)
            feat_attacked_node2 = np.random.choice(str_non_attacked_node, len(str_non_attacked_node)//2, replace=False)
            feat_attacked_nodes = np.sort(np.concatenate((feat_attacked_node1, feat_attacked_node2)))
            feat_attacked_mask = np.in1d(np.arange(dataset.x.size(0)), feat_attacked_nodes)
            perturb = torch.randn_like(dataset.x)
            dataset.x[feat_attacked_mask] = dataset.x[feat_attacked_mask] + perturb[feat_attacked_mask]*0.5
            torch.save(dataset.x, f'{root}{name}_{attack}_x.pt')
            print(f'# Only Str-Attacked Nodes: {(str_attacked_node_mask * ~feat_attacked_mask).sum()}, # Only Ft-Attacked Nodes: {(~str_attacked_node_mask * feat_attacked_mask).sum()}, # Both-Attacked Nodes: {(feat_attacked_mask*str_attacked_node_mask).sum()}, # Pure Nodes {(~feat_attacked_mask*~str_attacked_node_mask).sum()}')
        return [dataset]


def create_masks(data):
    """
    Splits data into training, validation, and test splits in a stratified manner if
    it is not already splitted. Each split is associated with a mask vector, which
    specifies the indices for that split. The data will be modified in-place
    :param data: Data object
    :return: The modified data
    """
    tr = 0.1
    vl = 0.1
    tst = 0.8
    if not hasattr(data, "val_mask"):
        _train_mask = _val_mask = _test_mask = None

        for i in range(20):
            labels = data.y.numpy()
            dev_size = int(labels.shape[0] * vl)
            test_size = int(labels.shape[0] * tst)

            perm = np.random.permutation(labels.shape[0])
            test_index = perm[:test_size]
            dev_index = perm[test_size:test_size + dev_size]

            data_index = np.arange(labels.shape[0])
            test_mask = torch.tensor(np.in1d(data_index, test_index), dtype=torch.bool)
            dev_mask = torch.tensor(np.in1d(data_index, dev_index), dtype=torch.bool)
            train_mask = ~(dev_mask + test_mask)
            test_mask = test_mask.reshape(1, -1)
            dev_mask = dev_mask.reshape(1, -1)
            train_mask = train_mask.reshape(1, -1)

            if _train_mask is None:
                _train_mask = train_mask
                _val_mask = dev_mask
                _test_mask = test_mask

            else:
                _train_mask = torch.cat((_train_mask, train_mask), dim=0)
                _val_mask = torch.cat((_val_mask, dev_mask), dim=0)
                _test_mask = torch.cat((_test_mask, test_mask), dim=0)
        
        data.train_mask = _train_mask.squeeze()
        data.val_mask = _val_mask.squeeze()
        data.test_mask = _test_mask.squeeze()
    
    elif hasattr(data, "val_mask") and len(data.val_mask.shape) == 1:
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T
        data.test_mask = data.test_mask.T
    
    else:  
        num_folds = torch.min(torch.tensor(data.train_mask.size())).item()
        data.train_mask = data.train_mask.T
        data.val_mask = data.val_mask.T
        if len(data.test_mask.size()) == 1: 
            data.test_mask = data.test_mask.unsqueeze(0).expand(num_folds, -1) 
        else:
            data.test_mask = data.test_mask.T

    return data


def index_to_mask(index, size = None):
    r"""Converts indices to a mask representation.

    Args:
        idx (Tensor): The indices.
        size (int, optional). The size of the mask. If set to :obj:`None`, a
            minimal sized output mask is returned.
    """
    index = index.view(-1)
    size = int(index.max()) + 1 if size is None else size
    mask = index.new_zeros(size, dtype=torch.bool)
    mask[index] = True
    return mask
