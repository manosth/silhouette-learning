import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupSparseAE(nn.Module):
    def __init__(
        self,
        input_size,
        group_size,
        num_groups,
        num_layers,
        group_tau,
        group_lambda,
        n_channels,
        W=None,
        use_fista=True
    ):
        super(GroupSparseAE, self).__init__()

        self.num_layers = num_layers
        self.num_groups = num_groups
        self.input_size = input_size
        self.group_size = group_size
        self.dict_size = self.num_groups * self.group_size
        self.group_lambda = group_lambda
        self.group_tau = group_tau
        self.use_fista = use_fista
        self.n_channels = n_channels

        # actual parameters of the model
        if W is None:
            W_list = []
            for channel in range(self.n_channels):
                W = torch.randn(self.dict_size, self.input_size)
                W = F.normalize(W, dim=-1)
                W_list.append(nn.Parameter(W))
        self.W_list = nn.ParameterList(W_list)

    def normalize(self):
        for idx in range(self.n_channels):
            self.W_list[idx].div_(self.W_list[idx].norm(dim=-1, keepdim=True))

    def activation(self, x, one_sided=True):
        # we will reshape the latent vector so that the groups are easy to compute
        x = x.view(-1, self.num_groups, self.group_size)

        # the group-sparse thresholding
        if one_sided:
            out = F.relu(1 - self.group_lambda * self.group_tau / x.norm(dim=-1, keepdim=True)) * F.relu(x)
        else:
            out = F.relu(1 - self.group_lambda * self.group_tau / x.norm(dim=-1, keepdim=True)) * x

        return out.view(-1, self.dict_size)

    def encoder(self, y):
        batch_size, device = y.shape[0], y.device
        y = y.view(batch_size, self.n_channels, -1).unsqueeze(-2)

        if self.use_fista:
            x = torch.zeros(batch_size, self.n_channels, self.dict_size, device=device)
            for channel in range(self.n_channels):
                x_old = torch.zeros(batch_size, self.dict_size, device=device)
                x_tmp = torch.zeros(batch_size, self.dict_size, device=device)
                t_old = torch.tensor(1.0, device=device)

                precomp_W = self.W_list[channel] @ self.W_list[channel].transpose(-1, -2)
                precomp_y = y[:, channel, :, :].squeeze() @ self.W_list[channel].transpose(-1, -2)
                for k in range(self.num_layers):
                    grad = x_tmp @ precomp_W - precomp_y
                    x_new = self.activation(x_tmp - grad * self.group_tau)
                    t_new = (1 + torch.sqrt(1 + 4 * torch.pow(t_old, 2))) / 2
                    x_tmp = x_new + ((t_old - 1) / t_new) * (x_new - x_old)
                    x_old, t_old = x_new, t_new
                x[:, channel, :] = x_new
        else:
            x = torch.zeros(batch_size, self.n_channels, self.dict_size, device=device)
            for channel in range(self.n_channels):
                precomp_W = self.W_list[channel] @ self.W_list[channel].transpose(-1, -2)
                precomp_y = y[:, channel, :, :].squeeze() @ self.W_list[channel].transpose(-1, -2)
                for idx in range(self.num_layers):
                    grad = x_tmp @ precomp_W - precomp_y
                    x[:, channel, :] = self.activation(x - self.group_tau * grad)
        return x

    def decoder(self, z):
        batch_size, device = z.shape[0], z.device

        x_hat = torch.zeros(batch_size, self.n_channels, 1, self.input_size, device=device)
        for channel in range(self.n_channels):
            x_hat[:, channel, 0, :] = z[:, channel, :] @ self.W_list[channel]
        return x_hat

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z).view(x.shape)
        return x_hat