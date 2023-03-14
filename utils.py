# system imports
import os
import time
from skimage import io

# pythom imports
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset, random_split

import torchvision
import torchvision.datasets as ds

def report_statistics(start, idx, total_len, val=0.0):
    current = time.time()
    total = current - start
    seconds = int(total % 60)
    minutes = int((total // 60) % 60)
    hours = int((total // 60) // 60)

    if idx == -1:
        print("")
        print(f"Total time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
    else:
        remain = (total_len - idx - 1) / (idx + 1) * total
        seconds_r = int(remain % 60)
        minutes_r = int((remain // 60) % 60)
        hours_r = int((remain // 60) // 60)
        print(f"progress: {(idx + 1) / total_len * 100:5.2f}%\telapsed: {hours:02d}:{minutes:02d}:{seconds:02d}\tremaining: {hours_r:02d}:{minutes_r:02d}:{seconds_r:02d}\tval: {val}", end="\r")

def load_mnist(datadir="./data_cache"):
    train_ds = ds.MNIST(root=datadir, train=True, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    test_ds = ds.MNIST(root=datadir, train=False, download=True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                             ]))
    def to_xy(dataset):
        Y = dataset.targets.long()
        X = dataset.data.view(dataset.data.shape[0], params.n_channels, 1, -1) / 255.0
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    mean_tr = X_tr.mean(dim=0)
    X_tr -= mean_tr
    X_te -= mean_tr
    return X_tr, Y_tr, X_te, Y_te

def load_cifar(datadir='./data_cache'):
    train_ds = ds.CIFAR10(root=datadir, train=True,
                           download=True, transform=None)
    test_ds = ds.CIFAR10(root=datadir, train=False,
                          download=True, transform=None)

    def to_xy(dataset):
        Y = torch.Tensor(np.array(dataset.targets)).long()
        X = torch.Tensor(np.transpose(dataset.data, (0, 3, 1, 2))).float() / 255.0  # [0, 1]
        X = torchvision.transforms.Grayscale()(X).view(X.shape[0], 1, -1)
        return X, Y

    X_tr, Y_tr = to_xy(train_ds)
    X_te, Y_te = to_xy(test_ds)
    mean_tr = X_tr.mean(dim=0)
    X_tr -= mean_tr
    X_te -= mean_tr
    return X_tr, Y_tr, X_te, Y_te

def make_loader(dataset, shuffle=True, batch_size=128, num_workers=4):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=True)

def init_weights(X_tr, Y_tr, params, mode="PCA_whole"):
    if mode == "PCA_whole":
        print(X_tr.shape)
        V = X_tr.svd().V
        P = torch.empty(params.n_channels, params.num_groups, params.group_size, params.group_size)
        nn.init.orthogonal_(P)
        W = P.view(params.n_channels, params.group_size * params.num_groups, params.group_size) @ V[:, :params.group_size].T
    elif mode == "PCA_class":
        W = torch.empty(params.n_channels, params.group_size * params.num_groups, params.input_size)
        for class_v in range(params.n_classes):
            X_class = X_tr[Y_tr == class_v]
            V = X_class.view(X_class.shape[0], -1).svd().V
            P = torch.empty(params.n_channels, params.group_size, params.group_size)
            nn.init.orthogonal_(P)
            W_dig = P @ V[:, :params.group_size].T
            W[params.group_size * digit:params.group_size * (digit + 1), :] = W_dig

            X = X_class[random_ind].view(X_class[random_ind].shape[0], X_class[random_ind].shape[1], -1)
            W[:, params.group_size * class_v:params.group_size * (class_v + 1), :] = X.transpose(0, 1)
    elif mode == "data_whole":
        random_ind = np.random.choice(X_tr.shape[0], params.group_size * params.num_groups)
        X = X_tr[random_ind].view(params.n_channels, params.group_size * params.num_groups, params.input_size)
        W = X
    elif mode == "data_class":
        W = torch.empty(params.n_channels, params.group_size * params.num_groups, params.input_size)
        for class_v in range(params.n_classes):
            X_class = X_tr[Y_tr == class_v]
            random_ind = np.random.choice(X_class.shape[0], params.group_size)
            X = X_class[random_ind].view(X_class[random_ind].shape[0], X_class[random_ind].shape[1], -1)
            W[:, params.group_size * class_v:params.group_size * (class_v + 1), :] = X.transpose(0, 1)
    elif mode == "random":
        W = torch.randn(params.group_size * params.num_groups, params.input_size)
    elif mode == "ortho":
        W = torch.empty(params.group_size * params.num_groups, params.input_size)
        nn.init.orthogonal_(W)
    W = W.div(torch.linalg.matrix_norm(W, ord=2).view(params.n_channels, 1, 1))
    return W
