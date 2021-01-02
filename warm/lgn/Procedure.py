"""

"""
import numpy as np
import torch
import utils
from parse import args


def BPR_train_original(dataset, Recmodel, opt):
    Recmodel.train()
    S = utils.bpr_neg_samp(dataset.trainUniqueUsers, dataset.trainDataSize,
                           support_dict=dataset.allPos, item_array=dataset.trainUniqueItems)
    users = torch.Tensor(S[:, 0]).long().to(args.device)
    posItems = torch.Tensor(S[:, 1]).long().to(args.device)
    negItems = torch.Tensor(S[:, 2]).long().to(args.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    # loss and update
    total_batch = len(users) // args.batch_size + 1
    aver_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) \
            in enumerate(utils.minibatch(users, posItems, negItems, batch_size=args.batch_size)):
        loss, reg_loss = Recmodel.bpr_loss(dataset.graph, batch_users, batch_pos, batch_neg)
        reg_loss = reg_loss * args.reg_lambda
        loss = loss + reg_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        aver_loss += loss.cpu().item()

    aver_loss = aver_loss / total_batch
    return aver_loss


def test(dataset, testDict, Recmodel):
    u_batch_size = args.test_batch
    Recmodel = Recmodel.eval()  # eval mode with no dropout
    max_K = max(args.topks)

    results = {'precision': np.zeros(len(args.topks)),
               'recall': np.zeros(len(args.topks)),
               'ndcg': np.zeros(len(args.topks))}

    with torch.no_grad():
        users = list(testDict.keys())
        users_list, rating_list, groundTrue_list, auc_record = [], [], [], []
        total_batch = len(users) // u_batch_size + 1

        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)  # batch user pos in train
            groundTrue = [testDict[u] for u in batch_users]  # batch user pos in test
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(args.device)
            rating = Recmodel.getUsersRating(dataset.graph, batch_users_gpu)  # batch user rating / have used sigmoid

            # mask pos in train
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)

            # get top k rating item index
            _, rating_K = torch.topk(rating, k=max_K)
            rating_K = rating_K.cpu().numpy()

            # compute auc
            rating = rating.cpu().numpy()
            aucs = [utils.AUC(rating[i], test_data) for i, test_data in enumerate(groundTrue)]
            auc_record.extend(aucs)

            # store batch
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K)
            groundTrue_list.append(groundTrue)

        # compute metric
        assert total_batch == len(users_list)
        pre_results = []
        for rl, gt in zip(rating_list, groundTrue_list):
            batch_ks_result = np.zeros((3, len(args.topks)), dtype=np.float32)
            for i, k in enumerate(args.topks):
                batch_ks_result[:, i] = np.array([utils.test_batch(rl[:, :k], gt)], dtype=np.float32)
            pre_results.append(batch_ks_result)
        for result in pre_results:
            results['precision'] += result[0]
            results['recall'] += result[1]
            results['ndcg'] += result[2]
        results['precision'] /= float(len(users))
        results['recall'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        results['auc'] = np.mean(auc_record)

        return results
