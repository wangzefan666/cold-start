import argparse
import torch
import multiprocessing
import pickle

parser = argparse.ArgumentParser(description="Go!")
parser.add_argument('--dataset', type=str, default='CiteULike',
                    help="available datasets: [CiteULike, LastFM, XING]")
parser.add_argument('--datadir', type=str, default='../../data/process/')
parser.add_argument('--batch_size', type=int, default=2048, help="the batch size for bpr loss training procedure")
parser.add_argument('--embed_dim', type=int, default=200, help="the embedding size of lightGCN")
parser.add_argument('--n_layers', type=int, default=3, help="the layer num of model")
parser.add_argument('--layer_size', nargs='?', default='[64, 64, 64]', help="sizes of every layer's rep.")
parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
parser.add_argument('--lr', type=float, default=0.001, help="the learning rate")
parser.add_argument('--reg_lambda', type=float, default=1e-4, help="the lambda coefficient for l2 normalization")
parser.add_argument('--dropout', type=int, default=0, help="using the dropout or not")
parser.add_argument('--drop_rate', type=float, default=0.5, help="the proportion of batch size for training procedure")
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--n_fold', type=int, default=1, help="the fold num used to split large adj matrix, like gowalla")
parser.add_argument('--test_batch', type=int, default=100, help="the batch size of users for testing")
parser.add_argument('--test_every_n_epochs', type=int, default=20, help="test every n epochs")
parser.add_argument('--topks', nargs='?', default="[20]", help="@k test list")
parser.add_argument('--tensorboard', type=int, default=0, help="enable tensorboard")
parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

# others
args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
args.n_cores = multiprocessing.cpu_count() // 2
args.layer_size = eval(args.layer_size)
args.topks = eval(args.topks)

para_dict = pickle.load(open(args.datadir + args.dataset + '/warm_dict.pkl', 'rb'))


