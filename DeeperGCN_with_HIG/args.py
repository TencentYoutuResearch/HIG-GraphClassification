import argparse
import uuid
import logging
import time
import os
import sys
from utils.logger import create_exp_dir
import glob

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

class ArgsInit(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='DeeperGCN')
        # dataset
        parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                            help='dataset name (default: ogbg-molhiv)')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='number of workers (default: 0)')
        parser.add_argument('--batch_size', type=int, default=32,
                            help='input batch size for training (default: 32)')
        parser.add_argument('--feature', type=str, default='full',
                            help='two options: full or simple')
        parser.add_argument('--add_virtual_node', action='store_true')
        # training & eval settings
        parser.add_argument('--use_gpu', action='store_true')
        parser.add_argument('--device', type=int, default=0,
                            help='which gpu to use if any (default: 0)')
        parser.add_argument('--epochs', type=int, default=300,
                            help='number of epochs to train (default: 300)')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='learning rate set for optimizer.')     
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--grad_clip', type=float, default=0.,
                            help='gradient clipping')

        # model
        parser.add_argument('--num_layers', type=int, default=3,
                            help='the number of layers of the networks')
        parser.add_argument('--mlp_layers', type=int, default=1,
                            help='the number of layers of mlp in conv')
        parser.add_argument('--hidden_channels', type=int, default=256,
                            help='the dimension of embeddings of nodes and edges')
        parser.add_argument('--block', default='res+', type=str,
                            help='graph backbone block type {res+, res, dense, plain}')
        parser.add_argument('--conv', type=str, default='gen',
                            help='the type of GCNs')
        parser.add_argument('--gcn_aggr', type=str, default='max',
                            help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
        parser.add_argument('--norm', type=str, default='batch',
                            help='the type of normalization layer')
        parser.add_argument('--num_tasks', type=int, default=1,
                            help='the number of prediction tasks')
        # learnable parameters
        parser.add_argument('--t', type=float, default=1.0,
                            help='the temperature of SoftMax')
        parser.add_argument('--p', type=float, default=1.0,
                            help='the power of PowerMean')
        parser.add_argument('--learn_t', action='store_true')
        parser.add_argument('--learn_p', action='store_true')
        parser.add_argument('--y', type=float, default=0.0,
                            help='the power of softmax_sum and powermean_sum')
        parser.add_argument('--learn_y', action='store_true')
        
        
        '''
        Args for Deep AUC Maximization
        '''
        parser.add_argument('--configs', type=str, default='', help='')   
        parser.add_argument('--model_name', type=str, default='deepgcn', help='')   
        parser.add_argument('--loss', type=str, default='auroc', help='')   
        parser.add_argument('--optimizer', type=str, default='pesg', help='')   
        parser.add_argument('--weight_decay', type=float, default=1e-4) 
        parser.add_argument('--random_seed', type=int, default=0) # try different seeds
        parser.add_argument('--pretrained', type=boolean_string, default=False) 
        parser.add_argument('--activations', type=str, default='relu')
        
        # AUC-margin loss
        parser.add_argument('--gamma', type=float, default=500)
        parser.add_argument('--margin', type=float, default=1.0)
        parser.add_argument('--imratio', type=float, default=0.01)
        
        
        # message norm
        parser.add_argument('--msg_norm', action='store_true')
        parser.add_argument('--learn_msg_scale', action='store_true')
        # encode edge in conv
        parser.add_argument('--conv_encode_edge', action='store_true')
        # graph pooling type
        parser.add_argument('--graph_pooling', type=str, default='mean',
                            help='graph pooling method')
        # save model
        parser.add_argument('--model_save_path', type=str, default='model_ckpt',
                            help='the directory used to save models')
        parser.add_argument('--save', type=str, default='EXP', help='experiment name')
        parser.add_argument('--save_dir', type=str, default='', help='experiment name')
        # load pre-trained model
        parser.add_argument('--model_load_path', type=str, default='ogbg_molhiv_pretrained_model.pth',
                            help='the path of pre-trained model')
        
        # for colab
        parser.add_argument('-f', type=str, default='kernel')       
        self.args = parser.parse_args()

    def save_exp(self):
        self.args.save = '{}-B_{}-C_{}-L_{}-F_{}-DP_{}' \
                    '-GA_{}-T_{}-LT_{}-P_{}-LP_{}-Y_{}-LY_{}' \
                    '-MN_{}-LS_{}-RS_{}'.format(self.args.save, self.args.block, self.args.conv,
                                          self.args.num_layers, self.args.hidden_channels,
                                          self.args.dropout, self.args.gcn_aggr,
                                          self.args.t, self.args.learn_t, self.args.p, self.args.learn_p,
                                          self.args.y, self.args.learn_y,
                                          self.args.msg_norm, self.args.learn_msg_scale, 
                                          self.args.random_seed)

        self.args.save_dir = './saved_models/{}'.format(self.args.save) #, time.strftime("%Y%m%d-%H%M%S")) #, str(uuid.uuid4()))
        self.args.model_save_path = os.path.join(self.args.save_dir, self.args.model_save_path)
        

        create_exp_dir(self.args.save_dir) #, scripts_to_save=glob.glob('*.py'))
        
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            format=log_format,
                            datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.args.save_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        return self.args
