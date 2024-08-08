import sys
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


"""
class VAE_Conv4(nn.Module):
    def __init__(self, K, input_dims, filter1, filter2, filter3, filter4):
        super(VAE_Conv4, self).__init__()

        self.input_dims = input_dims
        self.filter1 = filter1
        self.filter2 = filter2
        self.filter3 = filter3
        self.filter4 = filter4

        self.conv1 = nn.Conv2d(input_dims[0], self.filter1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.filter1, self.filter2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.filter2, self.filter3, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.filter3, self.filter4, 3, stride=1, padding=1)

        self.maxpool1 = torch.nn.MaxPool2d(2, return_indices=True)
        self.maxpool2 = torch.nn.MaxPool2d(2, return_indices=True)
        self.maxpool3 = torch.nn.MaxPool2d(2, return_indices=True)
        self.maxpool4 = torch.nn.MaxPool2d(2, return_indices=True)

        if 'neurons' not in locals():
            neurons, self.shape_pre_flatten = self.getNeuronNum(torch.zeros(input_dims).unsqueeze(0))

        self.q_fc_mu = nn.Linear(neurons, K)
        self.q_fc_sig = nn.Linear(neurons, K)

        self.p_fc_upsample = nn.Linear(K, neurons)

        self.p_unflatten = nn.Unflatten(-1, self.shape_pre_flatten)

        self.p_deconv4 = nn.ConvTranspose2d(self.filter4, self.filter3, 3, stride=1, padding=1)
        self.p_deconv3 = nn.ConvTranspose2d(self.filter3, self.filter2, 3, stride=1, padding=1)
        self.p_deconv2 = nn.ConvTranspose2d(self.filter2, self.filter1, 3, stride=1, padding=1)
        self.p_deconv1 = nn.ConvTranspose2d(self.filter1, input_dims[0], 3, stride=1, padding=1)

        self.maxunpool4 = torch.nn.MaxUnpool2d(2)
        self.maxunpool3 = torch.nn.MaxUnpool2d(2)
        self.maxunpool2 = torch.nn.MaxUnpool2d(2)
        self.maxunpool1 = torch.nn.MaxUnpool2d(2)

        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def infer(self, x):

        #print(x.shape)

        x = F.relu(self.conv1(x))
        x, ind1 = self.maxpool1(x)
        #print(x.shape)

        x = F.relu(self.conv2(x))
        x, ind2 = self.maxpool2(x)
        #print(x.shape)

        x = F.relu(self.conv3(x))
        x, ind3 = self.maxpool3(x)
        #print(x.shape)

        x = F.relu(self.conv4(x))
        x, ind4 = self.maxpool4(x)
        #print(x.shape)

        flat_x = torch.flatten(x, 1)
        #print(flat_x.shape)

        mu = self.q_fc_mu(flat_x)
        sig = self.q_fc_sig(flat_x)

        return mu, sig, ind1, ind2, ind3, ind4

    def encode(self, x):

        x = torch.unsqueeze(x, 0)
        x = F.relu(self.conv1(x))
        x,_ = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x,_  = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x,_ = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x,_ = self.maxpool4(x)
        flat_x = torch.flatten(x, 1)
        out = self.q_fc_mu(flat_x)

        return out

    def decode(self, zs, ind1, ind2, ind3, ind4):
        b, k = zs.size()

        s = F.relu(self.p_fc_upsample(zs))
        #print(s.shape)

        s = s.view([b] + list(self.shape_pre_flatten[1:]))
        #print(s.shape)

        s = self.maxunpool4(s, ind4)
        s = F.relu(self.p_deconv4(s))
        #print(s.shape)

        s = self.maxunpool3(s, ind3)
        s = F.relu(self.p_deconv3(s))
        #print(s.shape)

        s = self.maxunpool2(s, ind2)
        s = F.relu(self.p_deconv2(s))
        #print(s.shape)

        s = self.maxunpool1(s, ind1)
        s = F.relu(self.p_deconv1(s))
        #print(s.shape)

        mu_xs = s.view(b, -1)

        return mu_xs

    def elbo(self, x):
        mu, sig, ind1, ind2, ind3, ind4 = self.infer(x)
        zs = rsample(mu, sig)
        mu_xs = self.decode(zs, ind1, ind2, ind3, ind4)

        return log_p_x(x, mu_xs, self.log_sig_x.exp()) - kl_q_p(zs, mu, sig)

    def get_sample(self, x):
        x = x.reshape([1] + list(x.shape))

        mu, sig, ind1, ind2, ind3, ind4 = self.infer(x)
        zs = rsample(mu, sig)
        mu_xs = self.decode(zs, ind1, ind2, ind3, ind4)

        return mu_xs

    def getNeuronNum(self, x):
        x = F.relu(self.conv1(x))
        x,_ = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x,_ = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x,_ = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x,_ = self.maxpool4(x)
        flat_x = torch.flatten(x, 1)
        return flat_x.numel(), x.shape
"""

class Conv4_Encoder(nn.Module):
    def __init__(self, K, input_dims, filters=[32, 64, 128, 256]):
        super(Conv4_Encoder, self).__init__()

        self.input_dims = input_dims
        self.filter1 = filters[0]
        self.filter2 = filters[1]
        self.filter3 = filters[2]
        self.filter4 = filters[3]

        self.conv1 = nn.Conv2d(input_dims[0], self.filter1, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(self.filter1, self.filter2, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(self.filter2, self.filter3, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(self.filter3, self.filter4, 3, stride=1, padding=1)

        self.maxpool1 = torch.nn.MaxPool2d(2, return_indices=True)
        self.maxpool2 = torch.nn.MaxPool2d(2, return_indices=True)
        self.maxpool3 = torch.nn.MaxPool2d(2, return_indices=True)
        self.maxpool4 = torch.nn.MaxPool2d(2, return_indices=True)

        if 'neurons' not in locals():
            neurons, self.shape_pre_flatten = self.getNeuronNum(torch.zeros(input_dims).unsqueeze(0))

        self.q_fc_mu = nn.Linear(neurons, K)
        self.q_fc_sig = nn.Linear(neurons, K)

        self.p_fc_upsample = nn.Linear(K, neurons)

        self.p_unflatten = nn.Unflatten(-1, self.shape_pre_flatten)

        self.p_deconv4 = nn.ConvTranspose2d(self.filter4, self.filter3, 3, stride=1, padding=1)
        self.p_deconv3 = nn.ConvTranspose2d(self.filter3, self.filter2, 3, stride=1, padding=1)
        self.p_deconv2 = nn.ConvTranspose2d(self.filter2, self.filter1, 3, stride=1, padding=1)
        self.p_deconv1 = nn.ConvTranspose2d(self.filter1, input_dims[0], 3, stride=1, padding=1)

        self.maxunpool4 = torch.nn.MaxUnpool2d(2)
        self.maxunpool3 = torch.nn.MaxUnpool2d(2)
        self.maxunpool2 = torch.nn.MaxUnpool2d(2)
        self.maxunpool1 = torch.nn.MaxUnpool2d(2)

        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def infer(self, x):

        #print(x.shape)

        x = F.relu(self.conv1(x))
        x, ind1 = self.maxpool1(x)
        #print(x.shape)

        x = F.relu(self.conv2(x))
        x, ind2 = self.maxpool2(x)
        #print(x.shape)

        x = F.relu(self.conv3(x))
        x, ind3 = self.maxpool3(x)
        #print(x.shape)

        x = F.relu(self.conv4(x))
        x, ind4 = self.maxpool4(x)
        #print(x.shape)

        flat_x = torch.flatten(x, 1)
        #print(flat_x.shape)

        mu = self.q_fc_mu(flat_x)
        sig = self.q_fc_sig(flat_x)

        return mu, sig, ind1, ind2, ind3, ind4

    def encode(self, x):

        x = torch.unsqueeze(x, 0)
        x = F.relu(self.conv1(x))
        x,_ = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x,_  = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x,_ = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x,_ = self.maxpool4(x)
        flat_x = torch.flatten(x, 1)
        out = self.q_fc_mu(flat_x)

        return out

    def decode(self, zs, ind1, ind2, ind3, ind4):
        b, k = zs.size()

        s = F.relu(self.p_fc_upsample(zs))
        #print(s.shape)

        s = s.view([b] + list(self.shape_pre_flatten[1:]))
        #print(s.shape)

        s = self.maxunpool4(s, ind4)
        s = F.relu(self.p_deconv4(s))
        #print(s.shape)

        s = self.maxunpool3(s, ind3)
        s = F.relu(self.p_deconv3(s))
        #print(s.shape)

        s = self.maxunpool2(s, ind2)
        s = F.relu(self.p_deconv2(s))
        #print(s.shape)

        s = self.maxunpool1(s, ind1)
        s = F.relu(self.p_deconv1(s))
        #print(s.shape)

        mu_xs = s.view(b, -1)

        return mu_xs


    def getNeuronNum(self, x):
        x = F.relu(self.conv1(x))
        x,_ = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x,_ = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x,_ = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x,_ = self.maxpool4(x)
        flat_x = torch.flatten(x, 1)
        return flat_x.numel(), x.shape


class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class resnet_encoder_block(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class resnet_decoder_block(nn.Module):
    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

"""
class ResNet18Enc(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=16, channels=1):
        super().__init__()

        self.in_planes = 64
        self.nc = channels
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(self.nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)

        # self.linear_mu = nn.Linear(512, z_dim)
        # self.linear_sig = nn.Linear(512, z_dim)

        self.linear = nn.Linear(512, 2*z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):

        x = torch.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # mu = self.linear_mu(x)
        # sig = self.linear_sig(x)

        x = self.linear(x)

        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]

        return mu, logvar

class ResNet18Dec(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=16, nc=1):
        super().__init__()

        self.in_planes = 512
        self.nc = nc

        self.linear = nn.Linear(z_dim, 2048)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):

        x = self.linear(z)
        x = x.view(z.size(0), 512, 2, 2)
        x = F.interpolate(x, scale_factor=4)

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), self.nc, 128, 128)

        return x

class ResNet34Enc(nn.Module):
    def __init__(self, num_Blocks=[3, 4, 6, 3], z_dim=16, nc=1):
        super().__init__()

        self.in_planes = 64
        self.nc = nc
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(self.nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)

        # self.linear_mu = nn.Linear(512, z_dim)
        # self.linear_sig = nn.Linear(512, z_dim)

        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        x = self.linear(x)

        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]

        return mu, logvar

class ResNet34Dec(nn.Module):
    def __init__(self, num_Blocks=[3, 4, 6, 3], z_dim=16, nc=1):
        super().__init__()

        self.in_planes = 512
        self.nc = nc

        self.linear = nn.Linear(z_dim, 2048)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 2, 2)
        x = F.interpolate(x, scale_factor=4)

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), self.nc, 128, 128)

        return x

class VAE_ResNet34(nn.Module):
    def __init__(self, z_dim, nc):
        super().__init__()

        self.encoder = ResNet34Enc(z_dim=z_dim, nc=nc)
        self.decoder = ResNet34Dec(z_dim=z_dim, nc=nc)

        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def forward(self, x):

        mu, sig = self.encoder(x)
        z = VAE_ResNet34.rsample(mu, sig)
        r = self.decoder(z)

        return r, z

    def elbo(self, x, beta):
        mu, sig = self.encoder(x)
        # z = self.reparameterize(mean, logvar)
        z = VAE_ResNet34.rsample(mu, sig)
        r = self.decoder(z)

        return VAE.log_p_x(x, r, self.log_sig_x.exp()) - beta * VAE.kl_q_p(z, mu, sig)

    @staticmethod
    def rsample(mu, sig):
        b, k = mu.size()

        eps = torch.randn(b, k, device=mu.device)

        return mu + eps * torch.exp(sig)

    @staticmethod
    def log_p_x(_x, _r, _sig_x):
        b, n = _x.size()[:2]
        _x = _x.reshape(b, -1)
        _r = _r.reshape(b, -1)

        squared_error = (_x - _r) ** 2 / (2 * _sig_x ** 2)

        return -(squared_error + torch.log(_sig_x)).sum(dim=1).mean(dim=(0))

    @staticmethod
    def kl_q_p(_z, _mu, _sig):
        log_p = -0.5 * (_z ** 2)
        log_q = -0.5 * (_z - _mu) ** 2 / _sig.exp() ** 2 - _sig

        return (log_q - log_p).sum(dim=1).mean(dim=0)
"""

class ResNet_Encoder(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=16, channels=1):
        super().__init__()

        self.in_planes = 64
        self.nc = channels
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(self.nc, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(resnet_encoder_block, 64, num_Blocks[0], stride=1)
        self.layer2 = self.make_layer(resnet_encoder_block, 128, num_Blocks[1], stride=2)
        self.layer3 = self.make_layer(resnet_encoder_block, 256, num_Blocks[2], stride=2)
        self.layer4 = self.make_layer(resnet_encoder_block, 512, num_Blocks[3], stride=2)

        # self.linear_mu = nn.Linear(512, z_dim)
        # self.linear_sig = nn.Linear(512, z_dim)

        self.linear = nn.Linear(512, 2*z_dim)

    def make_layer(self, _encoder_block, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks - 1)
        layers = []
        for stride in strides:
            layers += [_encoder_block(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):

        x = torch.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)

        # mu = self.linear_mu(x)
        # sig = self.linear_sig(x)

        x = self.linear(x)

        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]

        return mu, logvar

class ResNet_Decoder(nn.Module):
    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=16, channels=1):
        super().__init__()

        self.in_planes = 512
        self.nc = channels

        self.linear = nn.Linear(z_dim, 2048)

        self.layer4 = self.make_layer(resnet_decoder_block, 256, num_Blocks[3], stride=2)
        self.layer3 = self.make_layer(resnet_decoder_block, 128, num_Blocks[2], stride=2)
        self.layer2 = self.make_layer(resnet_decoder_block, 64, num_Blocks[1], stride=2)
        self.layer1 = self.make_layer(resnet_decoder_block, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, channels, kernel_size=3, scale_factor=2)

    def make_layer(self, _decoder_block, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [_decoder_block(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):

        x = self.linear(z)
        x = x.view(z.size(0), 512, 2, 2)
        x = F.interpolate(x, scale_factor=4)

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), self.nc, 128, 128)

        return x

class VAE(nn.Module):
    def __init__(self, _Enc, _Dec):
        super().__init__()

        self.encoder = _Enc
        self.decoder = _Dec

        self.log_sig_x = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        mu, sig = self.encoder(x)
        z = VAE.rsample(mu, sig)
        r = self.decoder(z)

        return r, z

    def elbo(self, x):
        mu, sig = self.encoder(x)
        # z = self.reparameterize(mean, logvar)
        z = VAE.rsample(mu, sig)
        r = self.decoder(z)

        return VAE.log_p_x(x, r, self.log_sig_x.exp()) - VAE.kl_q_p(z, mu, sig)

    @staticmethod
    def rsample(mu, sig):
        b, k = mu.size()

        eps = torch.randn(b, k, device=mu.device)

        return mu + eps * torch.exp(sig)

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std)
        return epsilon * std + mean

    @staticmethod
    def log_p_x(_x, _r, _sig_x):
        b, n = _x.size()[:2]
        _x = _x.reshape(b, -1)
        _r = _r.reshape(b, -1)

        squared_error = (_x - _r) ** 2 / (2 * _sig_x ** 2)

        return -(squared_error + torch.log(_sig_x)).sum(dim=1).mean(dim=(0))

    @staticmethod
    def kl_q_p(_z, _mu, _sig):
        log_p = -0.5 * (_z ** 2)
        log_q = -0.5 * (_z - _mu) ** 2 / _sig.exp() ** 2 - _sig

        return (log_q - log_p).sum(dim=1).mean(dim=0)

