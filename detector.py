import numpy as np
from utils import get_data, set_cuda_device, config2string, ensure_dir
import os

class detector:
    def __init__(self, args):
        print('===',args.dataset, '===') 
        
        self.args = args
        self.device = f'cuda:{args.device}'
        set_cuda_device(args.device)

        self.data_home = f'./dataset/'
        self.data = get_data(self.data_home, args.dataset, args.attack, args.ptb_rate)[0]
        self.clean = get_data(self.data_home, args.dataset, args.attack, 0.0)[0]

        # save results
        self.config_str = config2string(args)
        
        # basic statistics
        self.args.in_dim = self.data.x.shape[1]
        self.args.n_class = self.data.y.unique().size(0)
        self.args.n_node = self.data.x.shape[0]
        self.embed_dim = args.layers[-1]

        # save path check
        ensure_dir(f'{args.save_dir}/saved_model/detecting/')
        ensure_dir(f'{args.save_dir}/summary_result/detecting/{args.detector}/bypass/{args.dataset}/{args.attack}/{self.args.ptb_rate}/')     

    def summary_result(self):
        
        assert self.train_result.keys() == self.val_result.keys() and self.train_result.keys() == self.test_result.keys()

        key = list(self.train_result.keys())

        result_path = f'{self.args.save_dir}/summary_result/detecting/{self.args.detector}/{self.args.detector}_{self.args.dataset}_{self.args.task}_{self.args.attack}.txt'
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