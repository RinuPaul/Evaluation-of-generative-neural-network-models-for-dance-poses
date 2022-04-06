import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
os.sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.controller import init, DiagGaussian
dropout_rate = 0.0
EPS = 1e-6


class CustomEncoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size, latent_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        # self.bn2 = nn.BatchNorm1d(hidden_size)


    def encode(self, x, c):
        h1 = self.dropout(F.elu(self.fc1(torch.cat((x, c), dim=1))))
        h2 = self.dropout(F.elu(self.fc2(torch.cat((x, h1), dim=1))))
        # h1 = self.dropout(F.elu(self.bn1(self.fc1(torch.cat((x, c), dim=1)))))
        # h2 = self.dropout(F.elu(self.bn2(self.fc2(torch.cat((x, h1), dim=1)))))
        s = torch.cat((x, h2), dim=1)
        # return torch.tanh(self.mu(s)), torch.tanh(self.logvar(s))
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        z_min = z.min()
        z_max = z.max()
        if z_min < -1e5:
            print("input frame shape {}".format(x.shape))
            print("conditional frame shape {}".format(c.shape))
            print("minimum value of input frame: {}".format(x.min()))
            print("minimum value of conditional frame: {}".format(c.min()))
            print("exceptional minimum value of z is {}".format(z_min))
            print("mu minimum {}".format(mu.min()))
            print("logvar minimum {}".format(logvar.min()))
        if z_max > 1e5:
            print("input frame shape {}".format(x.shape))
            print("conditional frame shape {}".format(c.shape))
            print("maximum value of input frame: {}".format(x.max()))
            print("maximum value of conditional frame: {}".format(c.max()))                        
            print("exceptional maximum value of z is {}".format(z_max))
            print("mu maximum {}".format(mu.max()))
            print("logvar maximum {}".format(logvar.max()))
        return z, mu, logvar


class CustomMixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
                F.dropout,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
                F.dropout,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
                None,
            ),
        ]

        for index, (weight, bias,  _, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)
            # self.register_parameter("bn" + index, bn_params)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            # nn.BatchNorm1d(gate_hsize).to('cuda:0'),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hsize, gate_hsize),
            # nn.BatchNorm1d(gate_hsize).to('cuda:0'),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        # for (weight, bias, activation) in self.decoder_layers:
        for (weight, bias, activation, dropout) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            # input_min = input.min()
            # input_max = input.max()
            # print("input min value: {} max value {}".format(input_min, input_max))
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            # out = batchnorm(out) if batchnorm is not None else out
            layer_out = activation(out) if activation is not None else out
            layer_out = dropout(layer_out, p=dropout_rate) if dropout is not None else layer_out
            # output_min = layer_out.min()
            # output_max = layer_out.max()
            # print("output min value: {} max value {}".format(output_min, output_max))
        return layer_out


class CustomPoseMixtureVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = CustomEncoder(*args)
        self.decoder = CustomMixedDecoder(*args, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            # return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
            denominator = self.data_max - self.data_min
            denominator[denominator < EPS] = EPS
            return 2 * (t - self.data_min) / denominator - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            # return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
            demoninator = self.data_max - self.data_min
            demoninator[demoninator < EPS] = EPS
            return (t + 1) * demoninator / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def encode(self, x, c):
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)