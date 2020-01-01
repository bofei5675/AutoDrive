from utils import *
from train import *
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
import torch
import argparse
import gc
from torchvision.transforms import ToPILImage, ToTensor, RandomRotation, RandomHorizontalFlip,\
    Compose, Resize
from models.model_hg import HourglassNet
import os
import time
def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-sd', '--save-dir', type=str, dest='save_dir', default='run/')
    args.add_argument('-m', '--model', type=str, dest='model_type', default='HG',
                      choices=['UNet', 'HG'])
    args.add_argument('-ns', '--n-stacks', type=int, dest='num_stacks', default=6)
    args.add_argument('-nc', '--n-classes',  type=int, dest='num_classes', default=8)
    args.add_argument('-nf', '--n-features', type=int, dest='num_features', default=256)
    args.add_argument('-bs', '--batch_size', type=int, dest='batch_size', default=2)
    args.add_argument('-e', '--epoch', type=int, dest='epoch', default=30)
    args.add_argument('-j', '--job-type', dest='job_type', default=1, type=int)
    return args.parse_args()

def main():
    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    args = parse_args()
    save_dir = args.save_dir + 'model_{}_'.format(args.model_type) + current_time + '/'
    train_images_dir = PATH + 'train_images/{}.jpg'
    test_images_dir = PATH + 'test_images/{}.jpg'
    train = pd.read_csv(PATH + 'train.csv')  # .sample(n=20).reset_index()
    test = pd.read_csv(PATH + 'sample_submission.csv')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_dir + 'config.txt', 'w') as f:
        f.write(str(args))
    # train = train.iloc[:50, :]
    df_train, df_dev = train_test_split(train, test_size=0.05, random_state=42)
    df_test = test
    # Augmentation
    transform = Compose([
        ToPILImage(),
        ToTensor()
    ])

    # Create dataset objects
    train_dataset = CarDataset(df_train, train_images_dir, training=True)
    dev_dataset = CarDataset(df_dev, train_images_dir, training=False)
    test_dataset = CarDataset(df_test, test_images_dir, training=False)
    BATCH_SIZE = args.batch_size

    # Create data generators - they will produce batches
    # transform not using yet
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
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
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=max(n_epochs, 10) * len(train_loader) // 3, gamma=0.1)
    history = pd.DataFrame()
    best_loss = 1e6
    for epoch in range(n_epochs):
        torch.cuda.empty_cache()
        gc.collect()
        train_loss = train_model(save_dir, model, epoch, train_loader, device, optimizer, exp_lr_scheduler, history,
                                 args.job_type)
        best_loss, eval_loss = evaluate_model(model, epoch, dev_loader, device, best_loss, save_dir, history)
        with open(save_dir + 'log.txt', 'a+') as f:
            line = 'Epoch: {}; Train loss: {:.4f}; Eval Loss: {:.4f}; Best eval loss: {:.4f}\n'.format(epoch,
                                                                                                       train_loss,
                                                                                                       eval_loss,
                                                                                                       best_loss)
            f.write(line)
        history.to_csv(save_dir + 'history.csv', index=False)
        

if __name__ == '__main__':
    main()