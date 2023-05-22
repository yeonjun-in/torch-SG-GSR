import numpy as np
import sys
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from deeprobust.graph.defense import GCN
from torch_geometric.utils import  to_dense_adj
from torch_geometric.nn import Node2Vec
from detector import detector
from utils import set_everything
from collections import defaultdict
import scipy.sparse as sp
from torch_sparse import SparseTensor
from sklearn.linear_model import LogisticRegression

class node2vec(detector):
    def __init__(self, args):
        detector.__init__(self, args)
        self.args = args
    
    def training(self):
        device = f'cuda:{self.args.device}'
        self.train_result, self.val_result, self.test_result = defaultdict(list), defaultdict(list), defaultdict(list)
        self.best_epochs = []
        
        for seed in range(self.args.seed_n):
            self.seed = seed
            set_everything(seed)
            
            data = self.data.clone()

            if self.args.dataset not in ['cora', 'citeseer', 'pubmed', 'polblogs', 'garden_0', 'garden_100']:
                data.train_mask, data.val_mask, data.test_mask = data.train_mask[seed, :], data.val_mask[seed, :], data.test_mask[seed, :]
            
            data = data.cuda()
            row, col = data.edge_index
            
            self.model = Node2Vec(data.edge_index, embedding_dim=128, walk_length=20,
                                    context_size=10, walks_per_node=10,
                                    num_negative_samples=1, p=1, q=1, sparse=True, num_nodes=data.x.size(0))
            
            num_workers = 0 if sys.platform.startswith('win') else 4
            self.model = self.model.cuda()
            loader = self.model.loader(batch_size=128, shuffle=True,
                                        num_workers=num_workers)
            self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
            
            best, best_epochs, cnt_wait = 0, 0, 0
            for epoch in range(1, self.args.epochs+1):

                self.model.train()
                total_loss = 0 
                for pos_rw, neg_rw in loader:
                    self.optimizer.zero_grad()
                    loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item()

                loss = total_loss / len(loader)

                with torch.no_grad():
                    self.model.eval()
                    z = self.model()
                    
                clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150).fit(z[data.train_mask].detach().cpu().numpy(), data.y[data.train_mask].detach().cpu().numpy())
                train_acc = clf.score(z[data.train_mask].detach().cpu().numpy(), data.y[data.train_mask].detach().cpu().numpy())
                val_acc = clf.score(z[data.val_mask].detach().cpu().numpy(), data.y[data.val_mask].detach().cpu().numpy())
                test_acc = clf.score(z[data.test_mask].detach().cpu().numpy(), data.y[data.test_mask].detach().cpu().numpy())

                
                print(f'||Seed {seed} || Epoch {epoch} || Train Acc {train_acc:.2f} Val Acc {val_acc:.2f} Test Acc {test_acc:.2f}')

                if val_acc > best:
                    best = val_acc
                    cnt_wait = 0
                    best_epochs = epoch
                    torch.save(self.model.state_dict(), '{}/saved_model/detecting/best_{}_{}_{}_seed{}.pkl'.format(self.args.save_dir, self.args.dataset, self.args.attack, self.args.detector, seed))
                else:
                    cnt_wait += self.args.verbose
                
                if cnt_wait == self.args.patience:
                    print('Early stopping!')
                    break           
        
            self.model.load_state_dict(torch.load('{}/saved_model/detecting/best_{}_{}_{}_seed{}.pkl'.format(self.args.save_dir, self.args.dataset, self.args.attack, self.args.detector, seed), map_location=f'cuda:{self.args.device}'))
            self.model.eval()
            with torch.no_grad():
                self.model.eval()
                z = self.model()

                clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=150).fit(z[data.train_mask].detach().cpu().numpy(), data.y[data.train_mask].detach().cpu().numpy())
                train_acc = clf.score(z[data.train_mask].detach().cpu().numpy(), data.y[data.train_mask].detach().cpu().numpy())
                val_acc = clf.score(z[data.val_mask].detach().cpu().numpy(), data.y[data.val_mask].detach().cpu().numpy())
                test_acc = clf.score(z[data.test_mask].detach().cpu().numpy(), data.y[data.test_mask].detach().cpu().numpy())

            self.train_result[f'POISON_{self.args.attack}_{self.args.ptb_rate}'].append(train_acc.item()/100)
            self.val_result[f'POISON_{self.args.attack}_{self.args.ptb_rate}'].append(val_acc.item()/100)
            self.test_result[f'POISON_{self.args.attack}_{self.args.ptb_rate}'].append(test_acc.item()/100)
            
            torch.save(z.detach().cpu(), f'./results/summary_result/detecting/{self.args.detector}/bypass/{self.args.dataset}/{self.args.attack}/{self.args.ptb_rate}/detect_n2v_emb_seed{seed}.pt')            

        self.summary_result()    

