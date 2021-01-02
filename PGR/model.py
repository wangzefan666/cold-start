import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
from os import path
import torch.nn.functional as F


class SinMLP(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(SinMLP, self).__init__()
        self.X = embs
        mlp_list = [torch.nn.BatchNorm1d(2 * self.X.shape[-1], momentum=0.1).to(device)]
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = 2 * self.X.shape[-1]
            else:
                pre_size = mlp_size

            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = 2 * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)

    def forward(self, ids):
        batch = ids.shape[0]
        X_u = self.X[ids[:, 0]][:, 0, :]
        X_i = self.X[ids[:, 1]][:, 0, :]
        pred_x = self.mlp(torch.cat([X_u, X_i], dim=-1))
        return pred_x


class ModStack(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(ModStack, self).__init__()
        self.X = embs
        self.mds = []
        for i in range(3):
            for j in range(3):
                self.mds.append((i, j))

        mlp_list = []
        self.u_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.i_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.lys = layer_pairs
        self.lys_bn = torch.nn.BatchNorm1d(len(self.lys), momentum=0.1).to(device)
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = len(layer_pairs)
            else:
                pre_size = mlp_size
            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = len(layer_pairs) * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)
        if mlp_layer == 1:
            self.mlp[0].weight = torch.nn.Parameter(
                torch.FloatTensor(np.array(
                    [[1.0 / (len(layer_pairs))] * (len(layer_pairs))])).to(device))

    def forward(self, ids):
        xu = [self.u_bn[_](self.X[ids[:, 0]][:, _, :]) for _ in range(3)]
        xi = [self.i_bn[_](self.X[ids[:, 1]][:, _, :]) for _ in range(3)]
        p_list = [torch.sum(xu[ly[0]] * xi[ly[1]], dim=1, keepdim=True) for ly in self.lys]
        pred = self.lys_bn(torch.cat(p_list, dim=-1))
        pred_x = self.mlp(pred)
        return pred_x


class EleStack(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(EleStack, self).__init__()
        self.X = embs
        self.mds = []
        for i in range(3):
            for j in range(3):
                self.mds.append((i, j))

        mlp_list = []
        self.u_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.i_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.lys = layer_pairs
        self.lys_bn = torch.nn.BatchNorm1d(len(self.lys) * self.X.shape[-1], momentum=0.1).to(device)
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = len(layer_pairs) * self.X.shape[-1]
            else:
                pre_size = mlp_size
            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = len(layer_pairs) * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)
        if mlp_layer == 1:
            self.mlp[0].weight = torch.nn.Parameter(
                torch.FloatTensor(np.array(
                    [[1.0 / (len(layer_pairs) * self.X.shape[-1])] * (len(layer_pairs) * self.X.shape[-1])])).to(
                    device))

    def forward(self, ids):
        xu = [self.u_bn[_](self.X[ids[:, 0]][:, _, :]) for _ in range(3)]
        xi = [self.i_bn[_](self.X[ids[:, 1]][:, _, :]) for _ in range(3)]
        p_list = [xu[ly[0]] * xi[ly[1]] for ly in self.lys]
        pred = self.lys_bn(torch.cat(p_list, dim=-1))
        pred_x = self.mlp(pred)
        return pred_x


class MLPStack(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(MLPStack, self).__init__()
        self.X = embs
        self.mds = []
        for i in range(3):
            for j in range(3):
                self.mds.append((i, j))

        mlp_list = []
        self.u_map = nn.ModuleList([
            torch.nn.Linear(self.X.shape[-1], self.X.shape[-1], bias=False).to(device),
            torch.nn.Linear(self.X.shape[-1], self.X.shape[-1], bias=False).to(device),
            torch.nn.Linear(self.X.shape[-1], self.X.shape[-1], bias=False).to(device),
        ])
        self.i_map = nn.ModuleList([
            torch.nn.Linear(self.X.shape[-1], self.X.shape[-1], bias=False).to(device),
            torch.nn.Linear(self.X.shape[-1], self.X.shape[-1], bias=False).to(device),
            torch.nn.Linear(self.X.shape[-1], self.X.shape[-1], bias=False).to(device),
        ])
        self.u_map_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.i_map_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.u_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.i_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.lys = layer_pairs
        self.lys_bn = torch.nn.BatchNorm1d(len(self.lys) * self.X.shape[-1], momentum=0.1).to(device)
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = len(layer_pairs) * self.X.shape[-1]
            else:
                pre_size = mlp_size
            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = len(layer_pairs) * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)
        if mlp_layer == 1:
            self.mlp[0].weight = torch.nn.Parameter(
                torch.FloatTensor(np.array(
                    [[1.0 / (len(layer_pairs) * self.X.shape[-1])] * (len(layer_pairs) * self.X.shape[-1])])).to(
                    device))

    def forward(self, ids):
        xu = [self.u_map_bn[_](self.u_map[_](self.u_bn[_](self.X[ids[:, 0]][:, _, :]))) for _ in range(3)]
        xi = [self.i_map_bn[_](self.i_map[_](self.i_bn[_](self.X[ids[:, 1]][:, _, :]))) for _ in range(3)]
        p_list = [xu[ly[0]] * xi[ly[1]] for ly in self.lys]
        pred = self.lys_bn(torch.cat(p_list, dim=-1))
        pred_x = self.mlp(pred)
        return pred_x


class FullStack(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(FullStack, self).__init__()
        self.X = embs
        self.mds = []
        for i in range(3):
            for j in range(3):
                self.mds.append((i, j))

        mlp_list = []
        self.u_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.i_bn = nn.ModuleList([
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device),
            torch.nn.BatchNorm1d(self.X.shape[-1], momentum=0.1).to(device)
        ])
        self.lys = layer_pairs
        self.lys_bn = torch.nn.BatchNorm1d((len(self.lys) + 6) * self.X.shape[-1], momentum=0.1).to(device)
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = (len(self.lys) + 6) * self.X.shape[-1]
            else:
                pre_size = mlp_size
            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = (len(self.lys) + 6) * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)
        if mlp_layer == 1:
            self.mlp[0].weight = torch.nn.Parameter(
                torch.FloatTensor(np.array(
                    [[1.0 / ((len(self.lys) + 6) * self.X.shape[-1])] * ((len(self.lys) + 6) * self.X.shape[-1])])).to(
                    device))

    def forward(self, ids):
        xu = [self.u_bn[_](self.X[ids[:, 0]][:, _, :]) for _ in range(3)]
        xi = [self.i_bn[_](self.X[ids[:, 1]][:, _, :]) for _ in range(3)]
        p_list = [xu[ly[0]] * xi[ly[1]] for ly in self.lys]
        p_list += xu
        p_list += xi
        pred_x = self.mlp(self.lys_bn(torch.cat(p_list, dim=-1)))
        return pred_x


class MLP(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(MLP, self).__init__()
        self.X = embs
        mlp_list = [torch.nn.BatchNorm1d(6 * self.X.shape[-1], momentum=0.1).to(device)]
        for i in range(mlp_layer - 1):
            if i == 0:
                pre_size = 6 * self.X.shape[-1]
            else:
                pre_size = mlp_size

            linear = torch.nn.Linear(pre_size, mlp_size, bias=True).to(device)
            if if_xavier:
                nn.init.xavier_uniform_(linear.weight)
            mlp_list.append(linear)
            mlp_list.extend([
                nn.BatchNorm1d(mlp_size, momentum=0.1).to(device),
                nn.LeakyReLU(),
                nn.Dropout(p=drop_rate)])
        if mlp_layer <= 1:
            pre_size = 6 * self.X.shape[-1]
        else:
            pre_size = mlp_size
        linear = torch.nn.Linear(pre_size, 1, bias=True).to(device)
        if if_xavier:
            torch.nn.init.xavier_uniform_(linear.weight)
        mlp_list.append(linear)
        self.mlp = torch.nn.Sequential(*mlp_list)

    def forward(self, ids):
        batch = ids.shape[0]
        X_u = self.X[ids[:, 0]].reshape([batch, -1])
        X_i = self.X[ids[:, 1]].reshape([batch, -1])
        pred_x = self.mlp(torch.cat([X_u, X_i], dim=-1))
        return pred_x


class WeiSum(nn.Module):
    def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
        super(WeiSum, self).__init__()
        self.X = embs
        self.mds = []
        self.w1 = nn.Linear(3, 1, bias=False).to(device)
        self.w1.weight = torch.nn.Parameter(torch.FloatTensor(np.array([0.33, 0.33, 0.33])).to(device))
        self.w2 = nn.Linear(3, 1, bias=False).to(device)
        self.w2.weight = torch.nn.Parameter(torch.FloatTensor(np.array([0.33, 0.33, 0.33])).to(device))

    def forward(self, ids):
        X_u = self.X[ids[:, 0]]
        X_i = self.X[ids[:, 1]]
        s = X_u.shape
        X_u = self.w1(X_u.permute([1, 2, 0]).reshape([s[1], -1]).permute([1, 0])).reshape([s[2], s[0]]).permute([1, 0])
        X_i = self.w2(X_i.permute([1, 2, 0]).reshape([s[1], -1]).permute([1, 0])).reshape([s[2], s[0]]).permute([1, 0])
        pred_x = torch.sum(X_u * X_i, dim=-1)
        return pred_x

# class WeiSum(nn.Module):
#     def __init__(self, embs, layer_pairs, mlp_size=64, mlp_layer=3, if_xavier=True, drop_rate=0.5, device='cpu'):
#         super(WeiSum, self).__init__()
#         self.X = embs
#         self.mds = []
#         self.w1 = torch.nn.Parameter(
#             torch.Tensor(np.ones(3)/3).reshape([1,3,1]).to(device), requires_grad=True)
#         self.w2 = torch.nn.Parameter(
#             torch.Tensor(np.ones(3)/3).reshape([1,3,1]).to(device), requires_grad=True)

#     def forward(self, ids):
#         X_u = self.X[ids[:,0]]
#         X_i = self.X[ids[:,1]]
#         s = X_u.shape
#         X_u = (X_u*self.w1).sum(1)
#         X_i = (X_i*self.w2).sum(1)
#         pred_x = torch.sum(X_u*X_i,dim=-1)
#         return pred_x
