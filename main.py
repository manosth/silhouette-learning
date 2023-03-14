# systme imports
import os
from datetime import datetime

# pythom imports
import numpy as np

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.datasets as ds

# file imports
from utils import *
from plot_utils import *
from model import GroupSparseAE
from config import Params

if __name__ == '__main__':
    params = Params()

    seed = 42
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    workers = max(4 * torch.cuda.device_count(), 4)

    current_date = datetime.now().strftime("%Y-%m-%d-T%H%M%S")
    comments = "_CIFAR10_groups=" + str(params.num_groups) + "_size=" + str(params.group_size) + "_lam=" + str(params.group_lambda) + "_init=" + str(params.init_mode) + "_batch=" + str(params.batch_size)
    if params.tensorboard:
        tensorboard_path = "./runs/{}".format(current_date)

    # download data
    X_tr, Y_tr, X_te, Y_te = load_cifar()
    train_dl = make_loader(TensorDataset(X_tr, Y_tr), batch_size=params.batch_size, num_workers=workers)
    test_dl = make_loader(TensorDataset(X_te, Y_te), batch_size=params.batch_size, num_workers=workers)

    model = GroupSparseAE(params.input_size, params.group_size, params.num_groups, params.num_layers, params.group_tau, params.group_lambda, params.n_channels).to(device)

    W = init_weights(X_tr, Y_tr, params, mode=params.init_mode)
    for channel in range(params.n_channels):
        model.W_list[channel].data = W[channel, :, :]
    model = model.to(device)

    opt = optim.Adam(model.parameters(), lr=params.lr, eps=params.eps)
    loss_func = torch.nn.MSELoss()

    # how many times to show stats per epoch
    times_per_epoch = 10
    report_period = len(train_dl) // times_per_epoch
    plot_period = 5

    # save the params
    if params.tensorboard:
        writer = SummaryWriter(tensorboard_path + comments)
        writer.add_text("params", str(vars(params)), global_step=0)

    # how many atoms per row for each group?
    nrow = 8
    dictionaries_path = "figs/dictionaries/"
    os.makedirs(dictionaries_path, exist_ok=True)
    save_dictionary(model, params, 0, nrow, dictionaries_path, save_atom=True)

    # set the parameters for the reporting
    total_train = params.epochs * (len(train_dl) + len(test_dl))

    train_errors = []
    test_errors = []

    start = time.time()
    local = time.localtime()
    print(f"Starting iterations...\t(Start time: {local[3]:02d}:{local[4]:02d}:{local[5]:02d})")
    for epoch in range(1, params.epochs + 1):
        net_loss = 0.0
        n_total = 0

        model.train()
        for idx, batch in enumerate(train_dl):
            x, y = batch[0].to(device), batch[1].to(device)

            out = model(x)
            loss = loss_func(x, out)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            net_loss += loss.item() * len(x)
            n_total += len(x)
            with torch.no_grad():
                model.normalize()
            if idx % report_period == 0:
                train_loss = net_loss / n_total
                curr_train = (epoch - 1) * (len(train_dl) + len(test_dl)) + idx
                report_statistics(start, curr_train, total_train, val=np.round(train_loss, 4))
        train_loss = net_loss / n_total


        if (epoch % plot_period) == 0:
            save_dictionary(model, params, epoch, nrow, dictionaries_path, save_atom=True)

        model.eval()
        net_loss = 0.0
        n_total = 0
        with torch.no_grad():
            for idx, batch in enumerate(test_dl):
                x, y = batch[0].to(device), batch[1].to(device)

                out = model(x)
                loss = loss_func(x, out)

                net_loss += loss.item() * len(x)
                n_total += len(x)

                if idx % report_period == 0:
                    test_loss = net_loss / n_total
                    curr_train = epoch * len(train_dl) + (epoch - 1) * len(test_dl) + idx
                    report_statistics(start, curr_train, total_train, val=np.round(test_loss, 4))
        test_loss = net_loss / n_total

        train_errors.append(train_loss)
        test_errors.append(test_loss)
        if params.tensorboard:
            writer.add_scalar("train_loss", train_loss, epoch + 1)
            writer.add_scalar("test_loss", test_loss, epoch + 1)
    report_statistics(start, -1, total_train)
    if params.tensorboard:
        writer.close()

    # save model for visualization afterwards
    os.makedirs("saved_models/", exist_ok=True)
    torch.save(model.state_dict(), "./saved_models/" + current_date + comments + ".pth")

    logs_path = "logs/"
    os.makedirs(logs_path, exist_ok=True)
    file = open(logs_path + "log_groups=" + str(params.num_groups) + "_size=" + str(params.group_size) + "_lam=" + str(params.group_lambda) + "_init=" + str(params.init_mode) + "_batch=" + str(params.batch_size) + ".txt", "w")

    file.write("# params\n")
    file.write(str(vars(params)) + "\n\n")

    file.write("# train errors\n")
    for train_err in train_errors[:-1]:
        file.write(str(train_err) + " ")
    file.write(str(train_errors[-1]) + "\n\n")

    file.write("# test errors\n")
    for test_err in test_errors[:-1]:
        file.write(str(test_err) + " ")
    file.write(str(test_errors[-1]) + "\n")
    file.close()
