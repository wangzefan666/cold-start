import utils
import torch
import time
import Procedure
import numpy as np
from parse import args
import model
from pprint import pprint
import dataloader
from parse import para_dict
import torch.optim as optim

pprint(vars(args))
utils.set_seed(args.seed)

# dataset
dataset = dataloader.Loader(path=args.datadir + args.dataset)

# model
n_users = para_dict['user_num']
m_items = para_dict['item_num']
Recmodel = model.LightGCN(n_users, m_items).to(args.device)

weight_file = utils.getFileName()
print(f"model will be save in {weight_file}")

# loss
opt = optim.Adam(Recmodel.parameters(), lr=args.lr)

# result
best_val = {'recall': np.array([0.0]),
            'precision': np.array([0.0]),
            'ndcg': np.array([0.0]),
            'auc': np.array([0.0])}


test_res = {}
start = time.time()
for epoch in range(args.epochs):
    aver_loss = Procedure.BPR_train_original(dataset, Recmodel, opt)  # train func

    if (epoch + 1) % args.test_every_n_epochs == 0:
        print(f'EPOCH[{epoch + 1}/{args.epochs}]')
        print(f"loss:{aver_loss:.3e}")
        print(f"Total time:{time.time() - start}")
        start = time.time()

        print("TEST")
        tmp = Procedure.test(dataset, para_dict['val_nb'], Recmodel)  # test func
        print('Current val result: {recall:', tmp['recall'], 'precision:', tmp['precision'],
              'ndcg:', tmp['ndcg'], 'auc:', tmp['auc'], '}')

        if np.sum(tmp['recall']) > sum(best_val['recall']):
            best_val = tmp
            print('Best val result: {recall:', best_val['recall'], 'precision:', best_val['precision'],
                  'ndcg:', best_val['ndcg'], 'auc:', best_val['auc'], '}')
            torch.save(Recmodel.state_dict(), weight_file)

            test_res = Procedure.test(dataset, para_dict['test_nb'], Recmodel)
            print('test result: {recall:', test_res['recall'], 'precision:', test_res['precision'],
                  'ndcg:', test_res['ndcg'], 'auc:', test_res['auc'], '}')

print('final result: {recall:', test_res['recall'], 'precision:', test_res['precision'],
      'ndcg:', test_res['ndcg'], 'auc:', test_res['auc'], '}')

# store embedding
Recmodel.load_state_dict(torch.load(weight_file))
Recmodel.eval()
users, items = Recmodel.computer(dataset.graph)  # average sum
embedding = np.zeros((Recmodel.num_users + Recmodel.num_items, args.embed_dim))
embedding[:Recmodel.num_users, :] = users.detach().cpu().numpy()
embedding[Recmodel.num_users:, :] = items.detach().cpu().numpy()
np.save(args.datadir + args.dataset + '/lgn.npy', embedding)
print('embedding of lgn is stored in ' + args.datadir + args.dataset + '/lgn.npy')

