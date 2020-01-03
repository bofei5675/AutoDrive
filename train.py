import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
from tqdm import tqdm#_notebook as tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
import os
from scipy.optimize import minimize
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms, utils
from efficientnet_pytorch import EfficientNet
from utils import imread, preprocess_image, get_mask_and_regr,\
    IMG_WIDTH, IMG_HEIGHT, MODEL_SCALE, load_my_state_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)


class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(2) == 1

        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0, flip=flip)
        img = np.rollaxis(img, 2, 0)
        # Get mask and regression maps
        mask, regr, heatmap = get_mask_and_regr(img0, labels, flip=False)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr, heatmap]


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, mid_ch,out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
            self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)

        self.conv = double_conv(mid_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #print(x1.shape, x2.shape)
        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh


class MyUNet(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = up(1282, 1282 + 1024, 512)
        self.up2 = up(512, 512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))
        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        feats = torch.cat([bg, feats, bg], 3)
        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)
        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x


# pixel wise focal loss
def focal_loss(pred, true, mask, alpha, beta):
    pos_loss = mask * (1 - pred) ** alpha * torch.log(pred + 1e-12)
    neg_loss = ((1 - true) ** beta) * (pred ** alpha) * torch.log(1 - pred + 1e-12)
    loss = pos_loss + neg_loss
    loss = loss.mean(0).sum()
    return loss

def mse_loss(pred, true):
    diff = (pred - true) ** 2
    diff = diff.mean(0).sum()
    return diff
def criterion(prediction, mask, regr, heatmap, size_average=True, loss_type='BCE', alpha=2, beta=4):
    '''
    Implement BCE and pixel-wise focal loss
    alpha/beta are from center net paper
    :param prediction:
    :param mask:
    :param regr:
    :param heatmap:
    :param size_average:
    :param is_focal_loss:
    :param alpha:
    :param beta:
    :return:
    '''
    # Binary mask loss
    pred_mask = torch.sigmoid(prediction[:, 0])
    if loss_type=='BCE':
        #    mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
        mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)
        mask_loss = -mask_loss.mean(0).sum()
    elif loss_type == 'FL': # focal loss
        mask_loss = focal_loss(pred_mask, heatmap, mask, alpha, beta)
    elif loss_type == 'MSE':
        mask_loss = mse_loss(pred_mask, heatmap)
    # Regression L1 loss
    pred_regr = prediction[:, 1:]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = mask_loss + regr_loss
    if not size_average:
        loss *= prediction.shape[0]
    return loss


import time


def train_model(save_dir, model, epoch, train_loader, device, optimizer, exp_lr_scheduler, history=None, args=None):
    model.train()
    total_loss = 0
    total_batches = len(train_loader)
    total_stacks = args.num_stacks
    stack_losses = np.zeros(total_stacks)
    for batch_idx, (img_batch, mask_batch, regr_batch, heatmap_batch) in enumerate(tqdm(train_loader)):
        # print('Train loop:', img_batch.shape)
        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)
        heatmap_batch = heatmap_batch.to(device)
        output = model(img_batch)
        if type(output) is list:
            loss = 0
            for idx, stack_output in enumerate(output):
                loss += criterion(stack_output, mask_batch, regr_batch, heatmap_batch,
                                  size_average=True, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta)
                stack_losses[idx] += loss.item()
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # exp_lr_scheduler.step()
        if batch_idx % 20 == 0 or batch_idx == total_batches - 1:
            with open(save_dir + 'log.txt', 'a+') as f:
                line = '{} | {} | Total Loss: {:.4f}, Stack Loss:{}\n'\
                    .format(batch_idx + 1, total_batches, total_loss / (batch_idx + 1), stack_losses / (batch_idx + 1))
                f.write(line)
    if history is not None:
        history.loc[epoch, 'train_loss'] = total_loss / len(train_loader)
    line = 'Train Epoch: {} \tLR: {:.6f}\tLoss: {:.6f}'.format(
        epoch,
        optimizer.state_dict()['param_groups'][0]['lr'],
        loss.data)
    print(line)
    return total_loss / total_batches


def save_model(model, dir, epoch):
    if not os.path.exists(dir):
        os.makedirs(dir)

    torch.save(model, dir + 'model_{}.pth'.format(epoch))


def evaluate_model(model, epoch, dev_loader, device, best_loss, save_dir, history=None, args = None):

    model.eval()
    total_loss = 0

    with torch.no_grad():
        stack_loss = np.zeros(args.num_stacks)
        for img_batch, mask_batch, regr_batch, heatmap_batch in dev_loader:
            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)
            heatmap_batch = heatmap_batch.to(device)
            output = model(img_batch)

            if type(output) is list:
                for idx, stack_output in enumerate(output):
                    loss = criterion(stack_output, mask_batch, regr_batch, heatmap_batch,
                                     size_average=True, loss_type=args.loss_type, alpha=args.alpha, beta=args.beta)
                    stack_loss[idx] += loss.item()
                total_loss += np.mean(stack_loss)
    total_loss /= len(dev_loader.dataset)
    stack_loss /= len(dev_loader.dataset)
    if total_loss < best_loss:
        best_loss = total_loss
        save_model(model, save_dir, epoch)
    if history is not None:
        history.loc[epoch, 'dev_loss'] = total_loss
    print('Dev loss: {:.4f}; Stack average loss: {}'.format(total_loss, stack_loss))
    return best_loss, loss.item()
