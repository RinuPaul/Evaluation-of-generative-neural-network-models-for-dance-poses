import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from thops import timesteps_fn, cat_feature, split_feature, sum_s
from modules import LSTM, GRU, InvertibleConv1x1, ActNorm2d, Permute2d, GaussianDiag

def nan_throw(tensor, name="tensor"):
        stop = False
        if ((tensor!=tensor).any()):
            print(name + " has nans")
            stop = True
        if (torch.isinf(tensor).any()):
            print(name + " has infs")
            stop = True
        if stop:
            print(name + ": " + str(tensor))
            #raise ValueError(name + ' contains nans of infs')

def f(in_channels, out_channels, hidden_channels, cond_channels, network_model, num_layers):
    #print("in",in_channels+cond_channels)
    #print("out",out_channels)
    if network_model=="LSTM":
        return LSTM(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    if network_model=="GRU":
        return GRU(in_channels + cond_channels, hidden_channels, out_channels, num_layers)
    if network_model=="FF":
        return nn.Sequential(
        nn.Linear(in_channels+cond_channels, hidden_channels), nn.ReLU(inplace=False),
        nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=False),
        LinearZeroInit(hidden_channels, out_channels))

class FlowStep(nn.Module):
    FlowCoupling = ["additive", "affine"]
    NetworkModel = ["LSTM", "GRU", "FF"]
    FlowPermutation = {
        "reverse": lambda obj, z, logdet, rev: (obj.reverse(z, rev), logdet),
        "shuffle": lambda obj, z, logdet, rev: (obj.shuffle(z, rev), logdet),
        "invconv": lambda obj, z, logdet, rev: obj.invconv(z, logdet, rev)
    }

    def __init__(self, in_channels, hidden_channels, cond_channels,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 network_model="LSTM",
                 num_layers=2,
                 LU_decomposed=False):
                 
        # check configures
        assert flow_coupling in FlowStep.FlowCoupling,\
            "flow_coupling should be in `{}`".format(FlowStep.FlowCoupling)
        assert network_model in FlowStep.NetworkModel,\
            "network_model should be in `{}`".format(FlowStep.NetworkModel)
        assert flow_permutation in FlowStep.FlowPermutation,\
            "float_permutation should be in `{}`".format(
                FlowStep.FlowPermutation.keys())
        super().__init__()
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.network_model = network_model
        # 1. actnorm
        self.actnorm = ActNorm2d(in_channels, actnorm_scale)
        # 2. permute
        if flow_permutation == "invconv":
            self.invconv = InvertibleConv1x1(
                in_channels, LU_decomposed=LU_decomposed)
        elif flow_permutation == "shuffle":
            self.shuffle = Permute2d(in_channels, shuffle=True)
        else:
            self.reverse = Permute2d(in_channels, shuffle=False)
        # 3. coupling
        if flow_coupling == "additive":
            self.f = f(in_channels // 2, in_channels-in_channels // 2, hidden_channels, cond_channels, network_model, num_layers)
        elif flow_coupling == "affine":
            #print("affine: in_channels = " + str(in_channels))
            self.f = f(in_channels // 2, 2*(in_channels-in_channels // 2), hidden_channels, cond_channels, network_model, num_layers)
            #print("Flowstep affine layer: " + str(in_channels))

    def init_lstm_hidden(self):
        if self.network_model == "LSTM" or self.network_model == "GRU":
            self.f.init_hidden()

    def forward(self, input, cond, logdet=None, reverse=False):
        if not reverse:
            return self.normal_flow(input, cond, logdet)
        else:
            return self.reverse_flow(input, cond, logdet)

    def normal_flow(self, input, cond, logdet):
    
        #assert input.size(1) % 2 == 0
        # 1. actnorm
        #z=input
        z, logdet = self.actnorm(input, logdet=logdet, reverse=False)
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, False)
        # 3. coupling
        z1, z2 = split_feature(z, "split")
        z1_cond = torch.cat((z1, cond), dim=1)
        if self.flow_coupling == "additive":            
            z2 = z2 + self.f(z1_cond)
            
        elif self.flow_coupling == "affine":
            #print("z1_cond:",z1_cond.shape)
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            #print("h:",h.shape)
            shift, scale = split_feature(h, "cross")
            #print(shift.shape,scale.shape,z2.shape)
            scale = torch.sigmoid(scale + 2.)+1e-6
            z2 = z2 + shift
            z2 = z2 * scale
            logdet = sum_s(torch.log(scale), dim=[1, 2]) + logdet
            #print(z1.shape,z2.shape)
        z = cat_feature(z1, z2)
        #print("z_step",z.shape)
        return z, cond, logdet

    def reverse_flow(self, input, cond, logdet):
        # 1.coupling
        z1, z2 = split_feature(input, "split")
        z1_cond = torch.cat((z1, cond), dim=1)

        if self.flow_coupling == "additive":
            z2 = z2 - self.f(z1_cond)
        elif self.flow_coupling == "affine":
            h = self.f(z1_cond.permute(0, 2, 1)).permute(0, 2, 1)
            shift, scale = split_feature(h, "cross")
            nan_throw(shift, "shift")
            nan_throw(scale, "scale")
            nan_throw(z2, "z2 unscaled")
            scale = torch.sigmoid(scale + 2.)+1e-6
            z2 = z2 / scale
            z2 = z2 - shift
            logdet = -sum_s(torch.log(scale), dim=[1, 2]) + logdet
            
        z = cat_feature(z1, z2)
        
        # 2. permute
        z, logdet = FlowStep.FlowPermutation[self.flow_permutation](
            self, z, logdet, True)
        nan_throw(z, "z permute_" + str(self.flow_permutation))
       # 3. actnorm
        z, logdet = self.actnorm(z, logdet=logdet, reverse=True)
        return z, cond, logdet


class FlowNet(nn.Module):
    def __init__(self, x_channels, hidden_channels, cond_channels, K,
                 actnorm_scale=1.0,
                 flow_permutation="invconv",
                 flow_coupling="additive",
                 network_model="LSTM",
                 num_layers=2,
                 LU_decomposed=False):
                 
        super().__init__()
        self.layers = nn.ModuleList()
        self.output_shapes = []
        self.K = K
        N = cond_channels
        for _ in range(K):
            self.layers.append(
                FlowStep(in_channels=x_channels,
                         hidden_channels=hidden_channels,
                         cond_channels=N,
                         actnorm_scale=actnorm_scale,
                         flow_permutation=flow_permutation,
                         flow_coupling=flow_coupling,
                         network_model=network_model,
                         num_layers=2,
                         LU_decomposed=LU_decomposed))
            self.output_shapes.append(
                [-1, x_channels, 1])

    def init_lstm_hidden(self):
        for layer in self.layers:
            if isinstance(layer, FlowStep):                
                layer.init_lstm_hidden()

    def forward(self, z, cond, logdet=0., reverse=False, eps_std=None):
        if not reverse:
            for layer in self.layers:
                z, cond, logdet = layer(z, cond, logdet, reverse=False)
                #print("z_net",z.shape)
            return z, logdet
        else:
            for i,layer in enumerate(reversed(self.layers)):
                z, cond, logdet = layer(z, cond, logdet=0, reverse=True)
            return z


class Glow(nn.Module):

    def __init__(self, x_channels, cond_channels):
        super().__init__()
        self.flow = FlowNet(x_channels=x_channels,
                            hidden_channels=512,
                            cond_channels=cond_channels,
                            K=16,
                            actnorm_scale=1.0,
                            flow_permutation="invconv",
                            flow_coupling="affine",
                            network_model="LSTM",
                            num_layers=2,
                            LU_decomposed=True)
        
        
        # register prior hidden
        #num_device = 1
        batch_size = 1
        self.z_shape = [batch_size , x_channels, 1]
        
        self.distribution = GaussianDiag()
        

    def init_lstm_hidden(self):
        self.flow.init_lstm_hidden()

    def forward(self, x=None, cond=None, z=None,
                eps_std=None, reverse=False):
        if not reverse:
            return self.normal_flow(x, cond)
        else:
            return self.reverse_flow(z, cond, eps_std)

    def normal_flow(self, x, cond):
    
        n_timesteps = timesteps_fn(x)

        logdet = torch.zeros_like(x[:, 0, 0])

        # encode
        z, objective = self.flow(x, cond, logdet=logdet, reverse=False)
        #print("z",z.shape)

        # prior
        objective += self.distribution.logp(z)
        

        # return
        nll = (-objective) / float(np.log(2.) * n_timesteps)
        return z, nll

    def reverse_flow(self, z, cond, eps_std):
        with torch.no_grad():

            z_shape = self.z_shape
            if z is None:
                z = self.distribution.sample(z_shape, eps_std, device=cond.device)

            x = self.flow(z, cond, eps_std=eps_std, reverse=True)
        return x

    def set_actnorm_init(self, inited=True):
        for name, m in self.named_modules():
            if (m.__class__.__name__.find("ActNorm") >= 0):
                m.inited = inited

    @staticmethod
    def loss_generative(nll):
        # Generative loss
        return torch.mean(nll)
        
