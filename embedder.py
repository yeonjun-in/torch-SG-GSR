import numpy as np
from utils import get_data, set_cuda_device, config2string, ensure_dir, to_numpy
import os
from copy import deepcopy
from datetime import datetime
from collections import defaultdict
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_undirected

class embedder:
    def __init__(self, args):
        print('===',args.dataset, '===') 
        assert args.cut_x + args.cut_z <= 0.7
        
        self.args = args
        self.device = f'cuda:{args.device}'
        set_cuda_device(args.device)

        self.data_home = f'./dataset/'
        self.data = get_data(self.data_home, args.dataset, args.attack, args.ptb_rate)[0]
        self.data.edge_adj = to_dense_adj(self.data.edge_index, max_num_nodes=self.data.x.size(0))[0]
        self.clean = get_data(self.data_home, args.dataset, 'clean', 0.0)[0]
        self.clean.edge_adj = to_dense_adj(self.clean.edge_index, max_num_nodes=self.clean.x.size(0))[0]
        
        # node feature knn graph
        self.x_sim = F.normalize(self.data.x, dim=1, p=2).mm(F.normalize(self.data.x, dim=1, p=2).T).fill_diagonal_(0.0)
        dst = self.x_sim.topk(self.args.knn, 1)[1]
        src = torch.arange(self.data.x.size(0)).unsqueeze(1).expand_as(dst)
        knn_edge = torch.stack([src.reshape(-1), dst.reshape(-1)])
        self.data.knn_edge = to_undirected(knn_edge)
            
        # save results
        self.config_str = config2string(args)
        self.train_result, self.val_result, self.test_result = defaultdict(list), defaultdict(list), defaultdict(list)

        # basic statistics
        self.args.in_dim = self.data.x.shape[1]
        self.args.n_class = self.data.y.unique().size(0)
        self.args.n_node = self.data.x.shape[0]

        # load path
        self.load_path = f'./results/summary_result/detecting/{self.args.best_detect}/bypass/{self.args.dataset}/{self.args.attack}/0.0/'
        self.detect_files = sorted([a for a in os.listdir(self.load_path) if 'detect' in a])
            
        ensure_dir(f'{args.save_dir}/saved_model/transfer/')
        ensure_dir(f'{args.save_dir}/summary_result/transferring/{args.embedder}/bypass/')

    def eval_base(self, data):
        train_mask, val_mask, test_mask = to_numpy(data.train_mask), to_numpy(data.val_mask), to_numpy(data.test_mask)
        
        with torch.no_grad():
            self.model.eval()
            logit = self.model.evaluate(data.x, data.edge_index)
            pred = logit.argmax(1)
        pred, y = to_numpy(pred), to_numpy(data.y)
        return pred, y, train_mask, val_mask, test_mask
    
    def verbose(self, data):
        pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
        correct = (pred==y)
        train_acc = np.mean(correct[train_mask])
        val_acc = np.mean(correct[val_mask])
        test_acc = np.mean(correct[test_mask])

        print(f'====== Train acc {train_acc*100:.2f}, Val acc {val_acc*100:.2f}, Test acc {test_acc*100:.2f},')
        return val_acc

    def eval_poisoning(self, data):
        save_dict = {'config':self.config_str}
        pred, y, train_mask, val_mask, test_mask = self.eval_base(data)
        correct = (pred==y)
        train_acc = np.mean(correct[train_mask])
        val_acc = np.mean(correct[val_mask])
        test_acc = np.mean(correct[test_mask])
        
        self.train_result[f'{self.args.attack}'].append(np.mean(train_acc))
        self.val_result[f'{self.args.attack}'].append(np.mean(val_acc))
        self.test_result[f'{self.args.attack}'].append(np.mean(test_acc))
        save_dict[f'{self.args.attack}'] = [pred, y, train_mask, val_mask, test_mask]
        torch.save(save_dict, f'{self.args.save_dir}/summary_result/transferring/{self.args.embedder}/bypass/{self.args.dataset}_{self.args.attack}_save_dict_poison_seed{self.seed}.pt')

    def summary_result(self):
        
        assert self.train_result.keys() == self.val_result.keys() and self.train_result.keys() == self.test_result.keys()

        key = list(self.train_result.keys())

        result_path = f'{self.args.save_dir}/summary_result/transferring/{self.args.embedder}/{self.args.embedder}_{self.args.dataset}_{self.args.task}_{self.args.attack}.txt'
        mode = 'a' if os.path.exists(result_path) else 'w'
        with open(result_path, mode) as f:
            f.write(self.config_str)
            f.write(f'\n')
            for k in key:
                f.write(f'====={k}=====')
                f.write(f'\n')
                train_mean, train_std = np.mean(self.train_result[k]), np.std(self.train_result[k])
                val_mean, val_std = np.mean(self.val_result[k]), np.std(self.val_result[k])
                test_mean, test_std = np.mean(self.test_result[k]), np.std(self.test_result[k])
                f.write(f'Train Acc: {train_mean*100:.2f}±{train_std*100:.2f}, Val Acc: {val_mean*100:.2f}±{val_std*100:.2f}, Test Acc: {test_mean*100:.2f}±{test_std*100:.2f}')
                f.write(f'\n')
                f.write(f'='*40)
                f.write(f'\n')