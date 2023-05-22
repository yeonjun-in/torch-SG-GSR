import pandas as pd
import numpy as np 
import os, gc, random, gzip, json
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_undirected, remove_self_loops
from torch_geometric.data import Data
from tqdm import tqdm
import argparse, gc
from scipy import sparse
from utils import index_to_mask

def parse_args():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--feat_attack", action='store_true', default=False)
    parser.add_argument('--min_user', default=1, type=int)
    parser.add_argument('--n_category', default=5, type=int)
    parser.add_argument('--min_item', default=5, type=int)
    parser.add_argument('--num_fraud', default=100, type=int)
    parser.add_argument('--num_attack', default=100, type=int)
    parser.add_argument('--edge_thre', default=2, type=int)
    parser.add_argument('--dataset_str', default='Automotive', type=str)
    parser.add_argument('--text', default='summary', type=str)
    parser.add_argument("--save", action='store_true', default=False)
    parser.add_argument('--device', default=0, type=int)

    return parser.parse_known_args()

args, _ = parse_args()
random.seed(1995)
os.environ['PYTHONHASHSEED'] = str(1995)
np.random.seed(1995)
torch.manual_seed(1995)
torch.cuda.manual_seed_all(1995)

device = f'cuda:{args.device}'
torch.cuda.set_device(device)

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield json.loads(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

asin, cate = [], []
with open(f'./fraud_rawdata/meta_{args.dataset_str}.json', 'r', encoding='UTF-8') as f:
   for line in tqdm(f):
        line = json.dumps(line)
        baby = eval(json.loads(line))
        asin.append(baby['asin'])
        cate.append(baby['categories'])
meta = pd.DataFrame({'asin':asin, 'category':cate})
review = getDF(f'./fraud_rawdata/reviews_{args.dataset_str}.json.gz')

meta['category'] = meta['category'].apply(lambda x: x[0])
meta = meta[meta['category'].apply(lambda x: len(x)) > 1]
meta['category'] = meta['category'].apply(lambda x: x[1])

df = meta[['asin', 'category']].merge(review, how='inner', on='asin')[['asin', 'category', 'reviewerID', 'helpful', args.text]] #reviewText
df.columns = ['asin', 'category', 'reviewerID', 'helpful', 'text']

del review, meta
gc.collect()

print('Preprocessing')
asin = df['asin'].value_counts()[lambda x: x>args.min_item].index
user = df['reviewerID'].value_counts()[lambda x: x>args.min_user].index
df = df[df.asin.isin(asin) & df.reviewerID.isin(user)]
df = df[df.category.isin(df.category.value_counts()[:args.n_category].index)].reset_index(drop=False)


print('Edge generation')
fraud = df['reviewerID'].value_counts()[lambda x: x==1].index.tolist()
fraud = np.random.choice(fraud, args.num_fraud, replace=False)
all_item = df['asin'].unique()

action_num = args.num_attack
user_list = []
item_list = []
for user in fraud:
    user_src = [user]*action_num
    fraud_action = np.random.choice(all_item, action_num, replace=True).tolist()
    user_list += user_src
    item_list += fraud_action

product_user_pair = pd.DataFrame({'asin':df['asin'].tolist() + item_list, 'reviewerID':df['reviewerID'].tolist()+user_list})
item2idx = {a:i for i,a in enumerate(product_user_pair.asin.unique())}
user2idx = {a:i for i,a in enumerate(product_user_pair.reviewerID.unique())}
product_user_pair.asin = product_user_pair.asin.map(item2idx)
product_user_pair.reviewerID = product_user_pair.reviewerID.map(user2idx)

I, J = product_user_pair.values.T
V = np.ones_like(I)
mat = sparse.coo_matrix((V,(I,J)),shape=(len(item2idx),len(user2idx)))
adj = (mat*mat.T)
adj.data *= adj.data >= args.edge_thre
adj.eliminate_zeros()

del product_user_pair, mat
gc.collect()

## node features 
print('Feature generation')
text = df.text.tolist()
item_id = df['asin'].tolist()
if args.feat_attack:
  # fake review generation
  fake_review = np.random.choice(text, len(item_list), replace=True).tolist()
  text = text + fake_review
  item_id = item_id + item_list
   

# Bag_of_word
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=300, binary=True)
vectorizer.fit(text)
X_bag_of_words = vectorizer.transform(text).toarray()

full_review_embedding_df = pd.DataFrame(X_bag_of_words).assign(item_id=item_id)
full_review_embedding_df['item_id'] = full_review_embedding_df['item_id'].map(item2idx)
grp = full_review_embedding_df.groupby('item_id').sum()
x = torch.from_numpy(grp.values)
x = (x>0).float()
grp_index = grp.index
del full_review_embedding_df, grp
gc.collect()


## node label 
print('Label generation')
df_ = df[['asin', 'category']].drop_duplicates()
df_['asin'] = df_['asin'].map(item2idx)
label = df_.set_index('asin').loc[grp_index.tolist()]
# label = df[['asin', 'category']].drop_duplicates().set_index('asin').loc[grp_index]
y = torch.from_numpy(pd.factorize(label['category'])[0]).long()

import torch.nn.functional as F
import torch
from torch_geometric.utils import is_undirected, to_undirected, to_dense_adj, remove_isolated_nodes, segregate_self_loops
# dataset = torch.load('dataset/fashion.pt')

edge_index = torch.from_numpy(np.stack(adj.nonzero())).long()
data = Data()
edge_index, _, _,_ = segregate_self_loops(edge_index)
edge_index, _, mask = remove_isolated_nodes(edge_index, num_nodes=x.size(0))
data.edge_index = edge_index
data.x = x[mask]
data.y = y[mask]
label_agree = (data.y.unsqueeze(1) == data.y.unsqueeze(0)).fill_diagonal_(0.0).numpy()
I, J = edge_index
V = np.ones_like(I)
new_adj = sparse.coo_matrix((V,(I,J)),shape=(data.x.size(0), data.x.size(0)))

print('fs')

random.seed(1995)
os.environ['PYTHONHASHSEED'] = str(1995)
np.random.seed(1995)
torch.manual_seed(1995)
torch.cuda.manual_seed_all(1995)

num_nodes = len(data.x)
num_label = int(num_nodes*0.1)
idx = torch.randperm(num_nodes)
idx_train = np.sort(idx[:num_label].numpy())
idx_val = np.sort(idx[num_label:num_label*2].numpy())
idx_test = np.sort(idx[num_label*2:].numpy())

##############GCN#############
from deeprobust.graph.defense import GCN
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from deeprobust.graph.defense import GCN
import scipy.sparse as sp
from deeprobust.graph import utils 
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

features = sp.csr_matrix(data.x.numpy().astype(np.float32))
labels = data.y.numpy().astype(np.int8)
model = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)
# using validation to pick model
model.fit(features, new_adj, labels, idx_train, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
output = model.predict()
pred = output.argmax(1).detach().cpu().numpy()
acc = accuracy_score(labels[idx_test], pred[idx_test])
bacc = balanced_accuracy_score(labels[idx_test], pred[idx_test])
f1 = f1_score(labels[idx_test], pred[idx_test], average='macro')

from utils import config2string
config_str = config2string(args)
result_path = f'./results/{args.dataset_str}_{args.text}.txt'
mode = 'a' if os.path.exists(result_path) else 'w'
with open(result_path, mode) as f:
    f.write(config_str)
    f.write(f'\n')
    f.write(f'A hom.: {0*100:.2f}, X hom.: {0*100:.2f}, N.Nodes: {num_nodes}, N.Edges: {data.edge_index.size(1)}')
    f.write(f'\n')
    f.write(f'GCN: Test Acc: {acc*100:.2f}, Test bACC: {bacc*100:.2f}, Test F1: {f1*100:.2f}')
    f.write(f'\n')
    
##############GCN#############
from mlp import MLP
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
import scipy.sparse as sp
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


labels = data.y.numpy().astype(np.int8)
model = MLP(nfeat=features.shape[1], nhid=16, nclass=labels.max()+1, device=device)
model = model.to(device)
# using validation to pick model
model.fit(features, new_adj, labels, idx_train, train_iters=200, verbose=True)
model.eval()
# You can use the inner function of model to test
output = model.predict()
pred = output.argmax(1).detach().cpu().numpy()
acc = accuracy_score(labels[idx_test], pred[idx_test])
bacc = balanced_accuracy_score(labels[idx_test], pred[idx_test])
f1 = f1_score(labels[idx_test], pred[idx_test], average='macro')

from utils import config2string
config_str = config2string(args)
result_path = f'./results/{args.dataset_str}_{args.text}.txt'
mode = 'a' if os.path.exists(result_path) else 'w'
with open(result_path, mode) as f:
    f.write(f'MLP: Test Acc: {acc*100:.2f}, Test bACC: {bacc*100:.2f}, Test F1: {f1*100:.2f}')
    f.write(f'\n')
    f.write(f'{y.unique(return_counts=True)[0]} {y.unique(return_counts=True)[1]}')
    f.write(f'\n')
    f.write(f'='*40)
    f.write(f'\n')

if args.save:
  print('Saving...')
  data.train_mask = index_to_mask(torch.from_numpy(idx_train), data.x.size(0))
  data.val_mask = index_to_mask(torch.from_numpy(idx_val), data.x.size(0))
  data.test_mask = index_to_mask(torch.from_numpy(idx_test), data.x.size(0))
  torch.save(data, f'dataset/{args.dataset_str}_{args.num_attack}{ "_feat_attack" if args.feat_attack else ""}.pt')