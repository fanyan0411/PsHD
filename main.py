import json
import argparse
from trainer import train
import ast

def main():
    args = setup_parser().parse_args()
    device = args.device
    memory_size_supervise = args.memory_size_supervise
    memory_size_unsupervise = args.memory_size_unsupervise
    memory_size = args.memory_size
    not_shuffle = args.not_shuffle
    model_name = args.model_name
    labeled_batch_size = args.labeled_batch_size
    batch_size = args.batch_size
    pse_threshold = args.pse_threshold
    seed = [args.seed]
    #print('seed', seed)
    param = load_json(args.config)
    args = vars(args)  # Converting argparse Namespace to a dict.
    prefix = args["prefix"]
    args.update(param)  # Add parameters from json
    if prefix != None:
        args["prefix"] = prefix
    
    #args["device"] = device
    args["seed"] = seed
    #args["memory_size_supervise"] = memory_size_supervise
    #args["memory_size_unsupervise"] = memory_size_unsupervise
    #args["memory_size"] = memory_size
    #args["toposimpl_dim"] = ast.literal_eval(args["toposimpl_dim"])
    #args["model_name"] = model_name
    #args["labeled_batch_size"] = labeled_batch_size
    #args["batch_size"] = batch_size
    print(args["labeled_batch_size"], args["batch_size"] )
    args["pse_threshold"] = pse_threshold
    if not_shuffle:
        args["shuffle"] = None
    
    train(args)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/der/der_10_300_uloss1.0_base_MS300_MU1700.json',
                        help='Json file of settings.')
    parser.add_argument('--model_name', type=str, default='der')
    parser.add_argument('--prefix', type=str, default='uloss1_base_psethre0.95_MS300_MU1700_20241120', 
                        help='prefix of log file')
    
    parser.add_argument('--label_num', type=str, default='300',
                        help='number of labeled data, need to be equal to prefix and bigger than memory_size_supervise') 
    
    parser.add_argument('--full_supervise', action='store_true')  #, default=True
    parser.add_argument('--batch_size', type=int, default=128)  # 512
    parser.add_argument('--labeled_batch_size', type=int, default=64)  # 256
    parser.add_argument('--init_epoch', type=int, default=200)
    parser.add_argument('--train_iterations', type=int, default=250)
    parser.add_argument('--less_train_iterations', type=int, default=50)
    ## Knowledge distillation
    parser.add_argument('--kd_weight', type=float, default=1, help='the weight of kd distill loss')
    parser.add_argument('--kd_onlylabel', action='store_true') #, default=True
    ## Memory setting
    parser.add_argument('--memory_size_supervise', type=int, default=300)
    parser.add_argument('--memory_size_unsupervise', type=int, default=1700)
    parser.add_argument('--memory_size', type=int, default=2000)
    ## Semi-supervised loss of unlabeled data
    parser.add_argument('--usp_weight',type=float, default=1, help='weight of semi-supervised loss')
    parser.add_argument('--insert_pse_progressive', action='store_true', help='adapt pseudo label of last model in semi-supervised loss')
    parser.add_argument('--insert_pse', type=str, help='threshold or logitstic', default='logitstic')
    parser.add_argument('--pse_weight', type=float, default=0.0, help='the weight of pseudo label in uloss')
    parser.add_argument('--oldpse_thre', type=float, default=0.8, help='the threshold of old pseudo label')
    parser.add_argument('--pse_threshold', default=0.95, type=float, help='confidence threshold in uloss') 
    ## Sub-graph distillation loss
    parser.add_argument('--rw_alpha', type=float, default=0, help='the weight of graph embedding') #TODO
    parser.add_argument('--match_weight', type=float, default=0.5, help='the weight of sub-graph distill')
    parser.add_argument('--rw_T', type=int, default=3, help='the order of structure embedding') #TODO
    parser.add_argument('--gamma_ml', type=float, default=1)
    ## Commom setting
    parser.add_argument('--resume', action='store_true', default=True) #, action='store_true' , type=bool, default=True
    parser.add_argument('--save_all_resume', action='store_true', default=False)
    parser.add_argument('--device', type=str, default=['3','4'])
    parser.add_argument('--not_shuffle', action='store_true') #, default=True

    ## Toposimplex setting
    parser.add_argument('--toposimpl_dim', type=str, default=[0,1])
    parser.add_argument('--embed_type',type=str,default="logits")
    parser.add_argument('--topo_weight', type=float, default=0.0, help='the weight of topo distill')
    parser.add_argument('--topo_weight_sample', type=float, default=None, help='the weight of sample-wise topo distill')
    parser.add_argument('--topo_weight_class', type=float, default=None, help='the weight of class-wise topo distill')
    parser.add_argument('--seed',type=str,default=1993)
    parser.add_argument('--save_resume_name',type=str,default=None)
    return parser


if __name__ == '__main__':
    main()
