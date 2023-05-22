import torch
import argparse
from utils import set_everything
import warnings 
warnings.filterwarnings("ignore")

def parse_args():
    # input arguments
    set_everything(1995)
    parser = argparse.ArgumentParser()    
    parser.add_argument('--task', default='transfer', choices=['detect', 'transfer'])
    parser.add_argument('--dataset', default='pubmed', choices=['cora', 'citeseer', 'pubmed', 'polblogs', 'computers', 'Patio_Lawn_and_Garden_0', 'Patio_Lawn_and_Garden_100', 'Pet_Supplies_0', 'Pet_Supplies_200'])
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--dropout', type = float, default=0.5)  

    if parser.parse_known_args()[0].task in ['detect']:
        parser.add_argument('--detector', default='node2vec')
        
    elif parser.parse_known_args()[0].task in ['transfer']:
        parser.add_argument('--embedder', default='SGGSR')
        parser.add_argument('--alpha', type = float, default=5)
        parser.add_argument('--cut_z', default=0.0, type=float)
        parser.add_argument('--dropedge', type = float, default=0.5)
        parser.add_argument('--cut_x', default=0.5, type=float)
        parser.add_argument('--topk', type = float, default = 0.1)
        parser.add_argument('--knn', default=5, type=int)
        parser.add_argument('--h_knn', default=5, type=int)
        parser.add_argument('--pos_ratio', default=1, type=float)
        parser.add_argument('--neg_ratio', default=0.5, type=float)
        
    parser.add_argument('--attack', type=str, default='noisy_str_meta', choices=['clean', 'clean_net', 'noisy_str_meta', 'noisy_str_net',  'real_world', 'real_world_net'])
    parser.add_argument('--ptb_rate', type=float, default=0.0)
    
    parser.add_argument('--seed_n', default=3, type=int)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument("--layers", nargs='*', type=int, default=[16])
    parser.add_argument('--heads', type = int, default = 8)

    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument("--no_debug", action='store_true', default=False)
    
    parser.add_argument('--best_detect', default='node2vec', type=str)
    
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--patience', type=int, default=400)    
    parser.add_argument('--verbose', type=int, default=1)    
    
    parser.add_argument('--save_dir', type=str, default='./results')

    return parser.parse_known_args()


def main():
    args, _ = parse_args()
        
    torch.cuda.set_device(args.device)
    if args.task == 'transfer':        
        if args.embedder == 'SGGSR':
            from models import SGGSR
            embedder = SGGSR(args)  
    
        embedder.training()
        print(embedder.config_str)
    
    elif args.task == 'detect':
        if args.detector == 'node2vec':
            from models_det import node2vec
            detector = node2vec(args) 
    
        detector.training()
        print(detector.config_str)

if __name__ == '__main__':
    main()
