import torch
import random, os
import numpy as np
from torch_scatter import scatter_add, scatter_mean, scatter_sum
from torch_sparse import SparseTensor, fill_diag, mul
from torch_sparse import sum as sparsesum
from torch_sparse import SparseTensor
from torch_sparse import spmm
import torch.nn.functional as F
from torch_geometric.utils import add_remaining_self_loops, dense_to_sparse, add_self_loops, to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.autograd import Variable

def jenson_shannon_divergence(net_1_logits, net_2_logits, reduction):
    net_1_probs = F.softmax(net_1_logits, dim=1)
    net_2_probs = F.softmax(net_2_logits, dim=1)
    
    total_m = 0.5 * (net_1_probs + net_2_probs)
    
    loss = 0.0
    loss += F.kl_div(F.log_softmax(net_1_logits, dim=1), total_m, reduction=reduction)
    loss += F.kl_div(F.log_softmax(net_2_logits, dim=1), total_m, reduction=reduction)
    return (0.5 * loss)



def reindex_edge_index(edge_index, sub_node):
    index_dict = {a:i for i, a in enumerate(sub_node.tolist())}
    src, dst = edge_index
    src = torch.LongTensor([index_dict[n] for n in src.tolist()])
    dst = torch.LongTensor([index_dict[n] for n in dst.tolist()])
    return torch.stack((src, dst))

def intersection_tensor(a,b):
    inter = set(to_numpy(a).tolist()).intersection(set(to_numpy(b).tolist()))
    return sorted(list(inter))




    
def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def set_cuda_device(device_num):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{device_num}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device) 

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        
        if name not in ['device', 'patience', 'epochs', 'task', 'save_dir', 'in_dim', 'n_class', 'best_epoch', 'save_fig', 'n_node', 'n_degree', 'verbose', 'mm', '', '' ,'sub_size', 'task_num', 'n_n', 'inner_steps', 'outer_steps', 'fine_epochs', 'sigma']:
            st_ = "{}:{} / ".format(name, val)
            st += st_
        
    
    return st[:-1]

def set_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # specify GPUs locally

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

