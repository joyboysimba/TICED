import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

import metric
from utils import collate_fn
from ticed import TICED
from dataset import load_data, RecSysDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='./datasets/yoochoose1_64/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
# parser.add_argument('--dataset_path', default='./DIDN/datasets/diginetica/', help='dataset directory path: datasets/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batch_size', type=int, default=64, help='input batch size 512')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for 100')
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate')
parser.add_argument('--lr_dc', type=float, default=0.07, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80,help='the number of steps after which the learning rate decay')
parser.add_argument('--test', default=False, help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--position_embed_dim', type=int, default=64, help='the dimension of position embedding')
parser.add_argument('--max_len', type=float, default=19, help='max length of input session')
parser.add_argument('--lambda_denoise', type=float, default=0.1, help='degree of lambda for denoising')
parser.add_argument('--pos_num', type=int, default=2000, help='the number of position encoding')
parser.add_argument('--neighbor_num', type=int, default=5, help='the number of neighboring sessions')
parser.add_argument('--num_heads', type=int, default=8, help='the head of MultiheadAttention')

args = parser.parse_args()
print(args)

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.cuda.set_device(6)


def main():
    print('Loading data...')
    train, valid, test = load_data(args.dataset_path, valid_portion=args.valid_portion)

    train_data = RecSysDataset(train)
    valid_data = RecSysDataset(valid)
    test_data = RecSysDataset(test)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.dataset_path.split('/')[-2] == 'diginetica':
        n_items = 43097
    elif args.dataset_path.split('/')[-2] == 'yoochoose1_64':
        n_items = 37484
    elif args.dataset_path.split('/')[-2] == 'Tmall':
        n_items = 40728
    else:
        n_items = 310
    model = TICED(n_items, args.hidden_size, args.embed_dim, args.batch_size, args.max_len, args.position_embed_dim, args.lambda_denoise, args.num_heads,args.pos_num, args.neighbor_num).to(device)

    if args.test:
        ckpt = torch.load('latest_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        p, mrr = validate(test_loader, model)
        print("Test: P@{}: {:.4f}, MRR@{}: {:.4f}".format(args.topk, p, args.topk, mrr))
        return

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
    best_result = [0, 0]
    for epoch in tqdm(range(args.epoch)):
        # train for one epoch
        scheduler.step(epoch=epoch)
        trainForEpoch(train_loader, model, optimizer, criterion, log_aggr=200)

        p, mrr = validate(valid_loader, model)
        print('Epoch {} validation: P@{}: {:.4f}, MRR@{}: {:.4f} \n'.format(epoch, args.topk, p, args.topk,
                                                                                 mrr))
        
        if p >= best_result[0]:
            best_result[0] = p
        if mrr >= best_result[1]:
            best_result[1] = mrr
        print('Best Result:')
        print('P@20:\t%.4f\tMRR@20:\t%.4f' % (best_result[0], best_result[1]))
        
        # store best loss and save a model checkpoint
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')


def trainForEpoch(train_loader, model, optimizer,criterion,log_aggr=200):
    model.train()

    sum_epoch_loss = 0

    for i, (seq, target, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        # neighbors = neighbors.to(device)

        optimizer.zero_grad()
        outputs = model(seq, lens)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        sum_epoch_loss += loss_val

        #if i % log_aggr == 0:
        #    print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
        #          % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
        #             len(seq) / (time.time() - start)))


def validate(valid_loader, model):
    model.eval()
    ps = []
    mrrs = []
    with torch.no_grad():
        for seq, target, lens in tqdm(valid_loader):
            seq = seq.to(device)
            target = target.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim=1)
            p, mrr = metric.evaluate(logits, target, k=args.topk)
            ps.append(p)
            mrrs.append(mrr)

    mean_p = np.mean(ps)
    mean_mrr = np.mean(mrrs)
    return mean_p, mean_mrr


class Set_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        cross_entropy = F.nll_loss(input, target)

        return cross_entropy


if __name__ == '__main__':
    main()
