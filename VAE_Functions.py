import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


def train_vae(vae, X_train, X_validate, epochs=50, batch_size=64, lr=1e-5, device='gpu'):

    if device == 'gpu':
        _device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        _device = 'cpu'

    opt = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=0)

    vae.to(_device)

    train_loader = DataLoader(X_train, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(X_validate, batch_size=batch_size, shuffle=True, pin_memory=True)

    train_losses = []
    validation_losses = []
    for ep in tqdm(range(epochs)):

        vae.train()
        for im in train_loader:
            # im_reconst, mu, logvar = vae.forward(im)
            # recon_loss = reconstruction_loss(im, im_reconst)
            # total_kld, _, _ = kl_divergence(mu, logvar)
            # train_loss = recon_loss + 1 * total_kld

            im = im.to(_device)
            opt.zero_grad()
            train_loss = -vae.elbo(im)
            train_loss.backward()
            opt.step()
            train_losses.append(-train_loss.item())

        vae.eval()
        for im in validation_loader:
            im = im.to(_device)
            val_loss = -vae.elbo(im)
            validation_losses.append(-val_loss.item())

    return train_losses, validation_losses

def Plot_Reconstruction(_img, _vae, _save_path=''):

    np.random.seed(seed=0)
    _idx = np.random.choice(range(len(_img)), size=50, replace=False)

    _X = _img[_idx]
    _X_i = torch.tensor(_X, dtype=torch.float).to(device)

    _vae.eval()
    _X_r, _ = _vae.forward(_X_i)
    _X_r = _X_r.cpu().detach()

    sph = 5
    spw = 10

    plt.figure(figsize=(24, 12))
    for i in range(len(_X)):
        plt.subplot(sph, spw, i + 1)
        plt.imshow(_X[i, 0], cmap='gray')
        plt.axis('off')

    plt.savefig(os.path.join(_save_path, 'original_images.png'), dpi=300)
    plt.clf()
    plt.close()

    plt.figure(figsize=(24, 12))
    for i in range(len(_X_r)):
        plt.subplot(sph, spw, i + 1)
        plt.imshow(_X_r[i, 0], cmap='gray')
        plt.axis('off')

    # plt.savefig(os.path.join(_save_path, 'reconstructed_images.png'), dpi=300)
    # plt.clf()
    # plt.close()

def Plot_History(_train_losses, _validation_losses, _save_path=''):

    _train_losses = _train_losses[3:]
    _validation_losses = _validation_losses[3:]

    _f = len(_train_losses) / len(_validation_losses)
    _fs = 7650

    plt.plot(np.arange(len(_train_losses)) / _fs, np.array(_train_losses), c='blue')
    plt.plot(_f * np.arange(len(_validation_losses)) / _fs, np.array(_validation_losses), c='orange')

    plt.xlabel('Epoch')
    plt.ylabel('ELBO')

    # plt.savefig(os.path.join(_save_path, 'Training_History.png'), dpi=300)

def Cycle_Colors(_n, color_num=None):
    # color_num: Number of colors through
    # n: Length of data

    if color_num == None:
        color_num = _n

    _colors = plt.cm.hsv(np.linspace(0, 1, color_num))
    _colors_rep = _colors.copy()
    for i in range(int(np.ceil(_n / color_num)) - 1):
        _colors_rep = np.concatenate((_colors_rep, _colors), axis=0)

    return _colors_rep

def plot_2D_hist(_X, _Y, colormap='jet', style='smooth'):
    H, ye, xe = np.histogram2d(_Y, _X, bins=1000)

    Hind = np.ravel(H)

    xc = (xe[:-1] + xe[1:]) / 2.0
    yc = (ye[:-1] + ye[1:]) / 2.0

    if style == 'smooth':
        sH = scipy.ndimage.gaussian_filter(H, sigma=10, order=0, mode='constant', cval=0.0)
        xv, yv = np.meshgrid(xc, yc)
        x_new = np.ravel(xv)[Hind != 0]
        y_new = np.ravel(yv)[Hind != 0]
        z_new = np.ravel(H if sH is None else sH)[Hind != 0]

    if style == 'absolute':
        xv, yv = np.meshgrid(xc, yc)
        x_new = np.ravel(xv)[Hind != 0]
        y_new = np.ravel(yv)[Hind != 0]
        z_new = Hind[Hind != 0]

    plt.scatter(x_new, y_new, c=z_new, s=1, cmap=colormap)

    if style == 'absolute':
        plt.colorbar()