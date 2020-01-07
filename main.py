from utils import *
from train import CarDataset, train_model, evaluate_model
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import torch
import argparse
import gc
from torchvision.transforms import ToPILImage, ToTensor, RandomRotation, RandomHorizontalFlip, \
    Compose, Resize, Normalize
from models.model_hg import HourglassNet
from models.model_hg2 import PoseNet
import os
import time
import torch.nn as nn
from albumentations import (
    RandomBrightnessContrast, Compose, RandomGamma, HueSaturationValue,
    RGBShift, MotionBlur, Blur, GaussNoise, ChannelShuffle
)

import warnings
warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-sd', '--save-dir', type=str, dest='save_dir', default='run/')
    args.add_argument('-m', '--model', type=str, dest='model_type', default='HG2',
                      choices=['UNet', 'HG', 'HG2'])
    args.add_argument('-ns', '--n-stacks', type=int, dest='num_stacks', default=8)
    args.add_argument('-nc', '--n-classes', type=int, dest='num_classes', default=8)
    args.add_argument('-nf', '--n-features', type=int, dest='num_features', default=256)
    args.add_argument('-bs', '--batch_size', type=int, dest='batch_size', default=2)
    args.add_argument('-e', '--epoch', type=int, dest='epoch', default=30)
    args.add_argument('-lf', '--loss-func', type=str,
                      dest='loss_type', default='BCE', choices=['BCE', 'FL', 'MSE'],
                      help='Loss function for supervising detection')
    args.add_argument('-a', '--alpha', type=int, dest='alpha', default=2)
    args.add_argument('-b', '--beta', type=int, dest='beta', default=4)
    args.add_argument('-db', '--debug', type=str2bool, dest='debug', default='no')
    args.add_argument('-s', '--sigma', type=int, dest='sigma', default=1)
    args.add_argument('-pt', '--pre-train', type=str2bool, dest='pre_train', default='yes')
    args.add_argument('-tp', '--transform-prob', type=float, dest='prob', default=0.2)
    args.add_argument('-g', '--gamma', type=float, dest='gamma', default=1, help='Weights for regression loss')
    args.add_argument('-vs', '--val-size', type=float, dest='val_size',
                      default=0.05, help='Validation data set size ratio')

    return args.parse_args()


def main():
    args = parse_args()
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    model_name = 'model_{}_stack_{}_features_{}_{}_' if args.prob <= 0 else 'model_aug_{}_stack_{}_features_{}_{}_'
    save_dir = args.save_dir + model_name.format(args.model_type, args.num_stacks, args.num_features, args.loss_type) \
               + current_time + '/'
    train_images_dir = PATH + 'train_images/{}.jpg'
    train = pd.read_csv(PATH + 'train_fixed.csv')  # .sample(n=20).reset_index()
    train = remove_out_image_cars(train)
    test = pd.read_csv(PATH + 'sample_submission.csv')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_dir + 'config.txt', 'w') as f:
        f.write(str(args))
    if args.debug:
        train = train.iloc[:50, :]
    df_train, df_dev = train_test_split(train, test_size=args.val_size, random_state=42)
    # Augmentation
    albu_list = [RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.3),
                 RandomGamma(p=0.2), HueSaturationValue(p=0.3), RGBShift(p=0.3), MotionBlur(p=0.1), Blur(p=0.1),
                 GaussNoise(var_limit=(20, 100), p=0.2),
                 ChannelShuffle(p=0.2),
                 Normalize(mean=[145.3834, 136.9748, 122.7390], std=[95.1996, 94.6686, 85.9170])]

    transform = Compose(albu_list, p=args.prob)

    # Create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, sigma=args.sigma, training=True, transform=transform)
    dev_dataset = CarDataset(df_dev, train_images_dir, sigma=args.sigma, training=False)
    BATCH_SIZE = args.batch_size

    # Create data generators - they will produce batches
    # transform not using yet
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # Gets the GPU if there is one, otherwise the cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print('Running on', torch.cuda.get_device_name())
    n_epochs = args.epoch
    if args.model_type == 'UNet':
        model = MyUNet(args.num_classes).to(device)
    elif args.model_type == 'HG':
        model = HourglassNet(nStacks=args.num_stacks, nModules=1, nFeat=args.num_features, nClasses=args.num_classes)
        model.cuda()
    elif args.model_type == 'HG2':
        model = PoseNet(nstack=args.num_stacks, inp_dim=args.num_features, oup_dim=args.num_classes)
        model = model.cuda()
        if args.num_stacks <= 2 and args.pre_train:
            save = torch.load('./weights/checkpoint_2hg.pt')
        elif args.pre_train:
            save = torch.load('./weights/checkpoint_8hg.pt')
        save = save['state_dict']
        # print(model)
        #  print(list(save.keys()))
        # print(model.state_dict().keys())
        load_my_state_dict(model, save)
        del save
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    if args.loss_type == 'FL':
        lr = 1e-3
    else:
        lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)
    history = pd.DataFrame()
    best_loss = 1e6
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train_loss, train_final_loss = train_model(save_dir, model, epoch, train_loader, device, optimizer,
                                                   exp_lr_scheduler, history,
                                                   args)
        best_loss, eval_loss, clf_losses, regr_losses = evaluate_model(model, epoch, dev_loader, device, best_loss,
                                                                       save_dir, history, args)
        with open(save_dir + 'log.txt', 'a+') as f:
            line = 'Epoch: {}; Train total loss: {:.3f}; Train final loss: {:.3f}; Eval final loss: {:.3f}; Clf loss: {:.3f}; Regr loss: {:.3f}; Best eval loss: {:.3f}\n' \
                .format(epoch,
                        train_loss,
                        train_final_loss,
                        eval_loss,
                        clf_losses,
                        regr_losses,
                        best_loss)
            f.write(line)
        history.to_csv(save_dir + 'history.csv', index=False)


if __name__ == '__main__':
    main()
