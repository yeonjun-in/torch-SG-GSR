import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from embedder import embedder
from encoder import MPCustom 
from utils import set_everything, intersection_tensor, reindex_edge_index, jenson_shannon_divergence
from torch_geometric.utils import negative_sampling, to_undirected, add_self_loops, subgraph, dropout_adj, degree, \
                                negative_sampling, coalesce, remove_self_loops, softmax, to_undirected, dense_to_sparse, is_undirected, batched_negative_sampling
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data
import math

class SGGSR(embedder):
    def __init__(self, args):
        if not args.no_debug:
            args.dataset, args.attack='pubmed', 'noisy_str_meta'
            args.lr, args.alpha, args.dropout = 0.001, 2, 0.6
            args.cut_z, args.cut_x = 0.0, 0.5
            args.topk = 0.3

        embedder.__init__(self, args)
        self.args = args

    def filter_noisy_edge(self, data):
        load_files = [a for a in self.detect_files if f'seed{self.seed}' in a][-1]
        z_sim = torch.load(self.load_path+load_files)
        if z_sim.size(0) != z_sim.size(1):
            z_sim = F.normalize(z_sim, dim=1, p=2).mm(F.normalize(z_sim, dim=1, p=2).T).fill_diagonal_(0.0).cpu()
        _, z_sim_score = dense_to_sparse(z_sim * data.edge_adj)
        _, x_sim_score = dense_to_sparse((self.x_sim+1) * data.edge_adj)
        _, i = z_sim_score.sort()
        z_remain_index = i[-int(data.edge_index.size(1) * (1-self.args.cut_z)):]
        _, i = x_sim_score.sort()
        x_remain_index = i[-int(data.edge_index.size(1) * (1-self.args.cut_x)):]
        final_remain_index = sorted(intersection_tensor(z_remain_index, x_remain_index))
        
        dst = z_sim.topk(self.args.h_knn, 1)[1]
        src = torch.arange(self.data.x.size(0)).unsqueeze(1).expand_as(dst)
        knn_edge = torch.stack([src.reshape(-1), dst.reshape(-1)])
        knn_edge = to_undirected(knn_edge)

        return final_remain_index, knn_edge


    def extract_clean_sub_graph(self, clean_edge, data_):
        sub_node = clean_edge.unique()
        sub_data = Data()
        sub_data.x = data_.x.clone()[sub_node]
        sub_data.edge_index = reindex_edge_index(clean_edge, sub_node)
        sub_data.y = data_.y.clone()[sub_node]
        sub_data.train_mask = data_.train_mask.clone()[sub_node]
        return sub_node, sub_data


    def training(self):
        self.best_epochs = []
        for seed in range(self.args.seed_n):
            self.seed = seed
            set_everything(seed)
            
            data = self.data.clone()
            if self.args.dataset not in ['cora', 'citeseer', 'pubmed', 'polblogs']:
                data.train_mask, data.val_mask, data.test_mask = data.train_mask[seed, :], data.val_mask[seed, :], data.test_mask[seed, :]

            # Define Clean edge
            final_remain_index, h_knn_edge = self.filter_noisy_edge(data)
            clean_link = to_undirected(data.edge_index[:, final_remain_index])
            
            ### Extract Sub-graph
            sub_node, sub_data = self.extract_clean_sub_graph(clean_link, data.clone())
            data.edge_adj = None

            # Complementary View
            h_knn_sub = subgraph(sub_node, h_knn_edge, relabel_nodes=True)[0]
            x_knn_sub = subgraph(sub_node, self.data.knn_edge, relabel_nodes=True)[0]
            sub_data.h_edge_index, sub_data.x_edge_index = h_knn_sub, x_knn_sub

            self.model = encoder(self.args)
            self.learner = learner(self.model, self.args)
            
            sub_data.main_view = sub_data.edge_index.clone()

            data = data.cuda()
            sub_data = sub_data.cuda()
            self.learner = self.learner.cuda()
            
            
            best, best_epochs, cnt_wait = 0, 0, 0
            for epoch in range(1, self.args.epochs+1):

                self.learner.train()
                losses = self.learner(sub_data)

                if epoch % self.args.verbose == 0:
                    with torch.no_grad():
                        self.learner.eval()
                        logit, out1, out2 = self.model(data.x, data.edge_index)
                        pred = logit.argmax(1)
                        correct = (pred == data.y).float()

                        train_acc = correct[data.train_mask].mean()*100
                        val_acc = correct[data.val_mask].mean()*100
                        test_acc = correct[data.test_mask].mean()*100
                        print(f'||S. {seed} || L. {losses.item():.4f} | Ep. {epoch} | Tr.A {train_acc:.2f} | Va.A {val_acc:.2f} | Te.A {test_acc:.2f} | B.Ep {best_epochs}')

                if val_acc > best:
                    best = val_acc
                    cnt_wait = 0
                    best_epochs = epoch
                    torch.save(self.model.state_dict(), f'{self.args.save_dir}/saved_model/transfer/best_model_{self.args.dataset}_{self.args.attack}_{self.args.ptb_rate}_{self.args.embedder}_seed{seed}.pkl')
                else:
                    cnt_wait += 1
                
                if cnt_wait == self.args.patience:
                    print('Early stopping!')
                    break           
                
                ### Edge addition augmentation
                if epoch > 10:
                    with torch.no_grad():
                        src, dst = sub_data.x_edge_index
                        dist = jenson_shannon_divergence(logit[sub_node][src], logit[sub_node][dst], reduction='none').mean(1)
                        values1, idx1 = (-dist).topk(int(sub_data.edge_index.size(1) * self.args.topk))
                        
                        src, dst = sub_data.h_edge_index
                        dist = jenson_shannon_divergence(logit[sub_node][src], logit[sub_node][dst], reduction='none').mean(1)
                        values2, idx2 = (-dist).topk(int(sub_data.edge_index.size(1) * self.args.topk))

                        if self.args.dataset == 'polblogs':
                            main_view = to_undirected(coalesce(torch.cat((sub_data.h_edge_index[:, idx2], sub_data.edge_index), dim=1)))
                        else:
                            main_view = to_undirected(coalesce(torch.cat((sub_data.x_edge_index[:, idx1], sub_data.h_edge_index[:, idx2], sub_data.edge_index), dim=1)))
                        sub_data.main_view = main_view
        
            self.model.load_state_dict(torch.load(f'{self.args.save_dir}/saved_model/transfer/best_model_{self.args.dataset}_{self.args.attack}_{self.args.ptb_rate}_{self.args.embedder}_seed{seed}.pkl', map_location=f'cuda:{self.args.device}'))
            self.eval_poisoning(data)
                
        self.summary_result()
        
class learner(torch.nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.wd)
        self.node_idx = torch.arange(args.n_node).cuda()
        self.args = args

    def forward(self, sub):
        
        deg = degree(sub.main_view[0])
        deg_q1, deg_q2 = deg.quantile(1/3), deg.quantile(2/3)
        
        src_degree, dst_degree = deg[sub.main_view[0]], deg[sub.main_view[1]]

        low_degree_src, low_degree_dst = src_degree <= deg_q1, dst_degree <= deg_q1
        mid_degree_src, mid_degree_dst = (deg_q1 < src_degree) & (deg_q2 >= src_degree), (deg_q1 < dst_degree) & (deg_q2 >= dst_degree)
        high_degree_src, high_degree_dst = src_degree > deg_q2, dst_degree > deg_q2

        low_low = low_degree_src * low_degree_dst
        mid_mid = mid_degree_src * mid_degree_dst
        high_high = high_degree_src * high_degree_dst

        low_mid = (low_degree_src * mid_degree_dst) | (mid_degree_src * low_degree_dst)
        low_high = (low_degree_src * high_degree_dst) | (high_degree_src * low_degree_dst)
        mid_high = (mid_degree_src * high_degree_dst) | (high_degree_src * mid_degree_dst)
        
        self.model.train()
        
        logit, out1, out2 = self.model(sub.x, dropout_adj(sub.main_view, p=self.args.dropedge)[0])
        ce_loss = F.cross_entropy(logit[sub.train_mask], sub.y[sub.train_mask])
        
        if low_low.sum() > 0:
            att_loss1_ll = self.attetion_loss_cal(out1, sub.main_view[:, low_low]).mean()
        else:
            att_loss1_ll = 0
        att_loss1_mm = self.attetion_loss_cal(out1, sub.main_view[:, mid_mid]).mean()
        att_loss1_hh = self.attetion_loss_cal(out1, sub.main_view[:, high_high]).mean()
        att_loss1_lm = self.attetion_loss_cal(out1, sub.main_view[:, low_mid]).mean()
        att_loss1_lh = self.attetion_loss_cal(out1, sub.main_view[:, low_high]).mean()
        att_loss1_mh = self.attetion_loss_cal(out1, sub.main_view[:, mid_high]).mean()

        if low_low.sum() > 0:
            att_loss2_ll = self.attetion_loss_cal(out2, sub.main_view[:, low_low]).mean()
        else:
            att_loss2_ll = 0
        
        att_loss2_mm = self.attetion_loss_cal(out2, sub.main_view[:, mid_mid]).mean()
        att_loss2_hh = self.attetion_loss_cal(out2, sub.main_view[:, high_high]).mean()
        att_loss2_lm = self.attetion_loss_cal(out2, sub.main_view[:, low_mid]).mean()
        att_loss2_lh = self.attetion_loss_cal(out2, sub.main_view[:, low_high]).mean()
        att_loss2_mh = self.attetion_loss_cal(out2, sub.main_view[:, mid_high]).mean()

        att_loss1 = att_loss1_ll + att_loss1_mm + att_loss1_hh + att_loss1_lm + att_loss1_lh + att_loss1_hh + att_loss1_mh
        att_loss2 = att_loss2_ll + att_loss2_mm + att_loss2_hh + att_loss2_lm + att_loss2_lh + att_loss2_hh + att_loss2_mh
        cost = ce_loss + att_loss1*self.args.alpha + att_loss2*self.args.alpha
        self.optimizer.zero_grad()
        cost.backward()
        self.optimizer.step()

        return cost

    def attetion_loss_cal(self, emb, pos):
        neg = self.model.negative_sampling(pos, pos, emb.size(0), None)
        pos_att = self.model.get_attention(edge_index_i=pos[1], x_i=emb[pos[1]], x_j=emb[pos[0]], 
                                        num_nodes=emb.size(0), return_logits=True, layer=1)
        neg_att = self.model.get_attention(edge_index_i=neg[1], x_i=emb[neg[1]], x_j=emb[neg[0]], 
                                        num_nodes=emb.size(0), return_logits=True, layer=1)
        att_x = torch.cat([pos_att, neg_att], dim=0)
        att_y = att_x.new_zeros(att_x.size(0))
        att_y[:pos_att.size(0)] = 1.
        att_loss = F.binary_cross_entropy_with_logits(att_x.mean(dim=-1), att_y, reduce=False)
        return att_loss

class encoder(MPCustom):    
    def __init__(self, args, concat=True, negative_slope=0.2,
                 add_self_loops=True, bias=True, attention_type='SD',
                 neg_sample_ratio=0.5, is_undirected=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim, self.hidden_size, self.out_dim, self.heads = args.in_dim, args.layers[0], args.n_class, args.heads
        self.args = args
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = self.args.dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.is_undirected = is_undirected 
        self.edge_sample_ratio = self.args.pos_ratio
        self.neg_sample_ratio=neg_sample_ratio

        assert attention_type in ['SD']
        assert 0.0 < self.neg_sample_ratio and 0.0 < self.edge_sample_ratio <= 1.0

        lin_1 = nn.Parameter(torch.FloatTensor(self.in_dim, self.heads * self.hidden_size))
        lin_2 = nn.Parameter(torch.FloatTensor(self.heads*self.hidden_size, self.heads*self.out_dim))
        
        self.att_x = self.att_y = None  # x/y for self-supervision

        if bias and concat:
            bias_1 = nn.Parameter(torch.FloatTensor(self.heads * self.hidden_size))
        elif bias and not concat:
            bias_1 = nn.Parameter(torch.FloatTensor(self.hidden_size))
        bias_2 = nn.Parameter(torch.FloatTensor(self.out_dim))
        
        self.vars = nn.ParameterList([
            lin_1, bias_1, lin_2, bias_2
        ])

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.vars[0]); glorot(self.vars[2])
        zeros(self.vars[1]); zeros(self.vars[3])

    def forward(self, x, edge_index, vars=None):
        x = F.dropout(x, p=self.args.dropout, training=self.training)
        
        if vars is None:
            vars = self.vars

        N, H, C1, C2 = x.size(0), self.heads, self.hidden_size, self.out_dim

        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        ######### Layer 1 ##########
        out1 = x.mm(vars[0]).view(-1, H, C1)
        out = self.propagate(edge_index, x=out1, size=None, vars=vars, layer=1) # attention weight 를 head 별로 구해서 aggregate 

        if self.concat is True:
            out = out.view(-1, self.heads * self.hidden_size)
        else:
            out = out.mean(dim=1)
        out += vars[1]
        out = F.elu(out)

        ######### Layer 2 ##########
        out = F.dropout(out, p=self.args.dropout, training=self.training)
        out2 = out.mm(vars[2]).view(-1, H, C2)
        out_ = self.propagate(edge_index, x=out2, size=None, vars=vars, layer=2)
        out_ = out_.mean(dim=1)
        out_ += vars[3]

        return out_, out1, out2

    def get_alpha(self):
        return

    def evaluate(self, x, edge_index):
        logit, _, _ = self.forward(x, edge_index)
        return logit

    def message(self, edge_index_i, x_i, x_j, size_i, vars, layer):
        
        alpha = self.get_attention(edge_index_i, x_i, x_j, num_nodes=size_i, vars=vars, layer=layer)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    def get_attention(self, edge_index_i, x_i, x_j, num_nodes, return_logits=False, vars=None, layer=1):
        
        out_channels = self.hidden_size if layer==1 else self.out_dim
        alpha = (x_i * x_j).sum(dim=-1) / math.sqrt(out_channels)
        if return_logits:
            return alpha

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        return alpha

    def negative_sampling(self, edge_index, pos_edge, num_nodes, batch=None):

        num_neg_samples = int(self.neg_sample_ratio * self.edge_sample_ratio *
                              pos_edge.size(1))

        if not self.is_undirected and not is_undirected(
                edge_index, num_nodes=num_nodes):
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        if batch is None:
            neg_edge_index = negative_sampling(edge_index, num_nodes,
                                               num_neg_samples=num_neg_samples)
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index, batch, num_neg_samples=num_neg_samples)

        return neg_edge_index


    def get_attention_loss(self, layer):
        if not self.training:
            return torch.tensor([0]).cuda()

        if layer==1:
            return F.binary_cross_entropy_with_logits(
                self.att_x_1.mean(dim=-1),
                self.att_y_1, reduce=False
            )
        elif layer==2:
            return F.binary_cross_entropy_with_logits(
                self.att_x_2.mean(dim=-1),
                self.att_y_2, reduce=False
            )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'type={self.attention_type})')


