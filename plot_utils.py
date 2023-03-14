# system imports
import os

# pythom imports
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm, rc


rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
sns.set_theme()
sns.set_context('paper')
sns.set(font_scale=1.4)
cmap = sns.cubehelix_palette(as_cmap=True)
color_plot = sns.cubehelix_palette(4)[1]

# torch imports
import torch
import torch.nn as nn
from torchvision.utils import make_grid, save_image

def save_dictionary(model, params, epoch, nrow, path, save_atom=False):
    matrix_data = torch.zeros(params.n_channels, params.num_groups * params.group_size, params.input_size)
    for channel in range(params.n_channels):
        matrix_data[channel, :, :] = model.W_list[channel].data.clone().detach().cpu()
    matrix_data = matrix_data.view(params.n_channels, params.num_groups, params.group_size, params.input_size)

    if params.group_size == 1:
        matrix_data = matrix_data.squeeze(2)
        matrix_data -= matrix_data.min()
        matrix_data /= matrix_data.max()
        grid = make_grid(
            matrix_data.transpose(0, 1).view(matrix_data.shape[1], params.n_channels, params.input_width,  params.input_height), nrow=40, padding=0
        )

        plt.figure(figsize=[14.4, 14.4])
        plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.savefig(path + "dict_groups=" + str(params.num_groups) + "_size=" + str(params.group_size) + "_lam=" + str(params.group_lambda) + "_init=" + str(params.init_mode) + "_batch=" + str(params.batch_size) + "_epoch=" + str(epoch) + ".pdf")
        plt.close()
    else:
        # make a grid for each group
        groupped_atoms = []
        for idx in range(matrix_data.shape[1]):
            group = matrix_data[:, idx, :, :]
            group -= group.min()
            group /= group.max()

            groupped_atoms.append(
                make_grid(
                    group.transpose(0, 1).view(group.shape[1], params.n_channels, params.input_width, params.input_height), nrow=nrow, padding=0
                )
            )

        fig, axs = plt.subplots(1, params.num_groups, figsize=[19.2, 4.8], constrained_layout=True)
        for group in range(params.num_groups):
            # matplotlib expects the channel information to be last
            axs[group].imshow(groupped_atoms[group].permute(1, 2, 0).numpy())
            #axs[group].title.set_text("group" + str(group))
            axs[group].title.set_text(params.classes[group])
            axs[group].axis('off')
        fig.suptitle("dictionary at epoch" + str(epoch))
        fig.savefig(path + "dict_groups=" + str(params.num_groups) + "_size=" + str(params.group_size) + "_lam=" + str(params.group_lambda) + "_init=" + str(params.init_mode) + "_batch=" + str(params.batch_size) + "_epoch=" + str(epoch) + ".pdf")
        plt.close(fig)

def save_final_means(classes, params, path):
    for class_v in range(params.n_classes):
        plt.figure()
        plt.bar(np.arange(params.group_size * params.num_groups), np.mean(classes[-1][class_v].detach().cpu().numpy(), axis=0), color=color_plot, edgecolor=color_plot)
        plt.ylabel("Magnitude")
        plt.xlabel("Index")
        plt.autoscale()
        plt.savefig(path + "means_class=" + str(params.classes[class_v]) + "_groups=" + str(params.num_groups) + "_size=" + str(params.group_size) + "_lam=" + str(params.group_lambda) + "_testlam=" + str(params.test_lambda) + "_int=" + str(params.init_mode) + "_batch=" + str(params.batch_size) + ".pdf", bbox_inches="tight")
        plt.close()

def save_pairwise(digits, params, num_exam, path):
    codes = []
    for digit in range(10):
        random_set = np.random.choice(len(digits[-1][digit]), num_exam, replace=False)
        codes.append(digits[-1][digit][random_set])
    codes = torch.cat(codes)

    codes = codes.view(codes.shape[0], params.num_groups, params.group_size)
    codes = codes.norm(dim=2)
    codes = codes.detach().cpu()

    D = codes.view(codes.shape[0], 1, -1) - codes.view(1, codes.shape[0], -1)
    D = D.norm(dim=-1).numpy()

    plt.figure()
    sns.color_palette("viridis")
    sns.heatmap(D)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.tight_layout(pad=0.2, w_pad=0.0, h_pad=0)
    plt.savefig(path + "pariwisedist_groups=" + str(params.num_groups) + "_size=" + str(params.group_size) + "_lam=" + str(params.group_lambda) + "_int=" + str(params.init_mode) + "_batch=" + str(params.batch_size) + "_testlam=" + str(params.test_lambda) + ".pdf")
    plt.close()
