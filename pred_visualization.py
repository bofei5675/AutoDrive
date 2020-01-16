import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm#_notebook as tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import coords2str, extract_coords
from train import CarDataset
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-lm', '--load-model', type=str, dest='load_model',
                      default=None)
    args.add_argument('-t', '--threshold', type=float, dest='threshold',
                      default=0)

    return args.parse_args()

def main():
    args = parse_args()
    print('Loading ...')
    PATH = './data/'
    test = pd.read_csv(PATH + 'sample_submission.csv')
    train = pd.read_csv(PATH + 'train.csv')  # .sample(n=20).reset_index()
    train_images_dir = PATH + 'train_images/{}.jpg'
    test_images_dir = PATH + 'test_images/{}.jpg'
    df_test = test
    test_dataset = CarDataset(train, train_images_dir, training=False)
    load_model = args.load_model
    save_dir = load_model.split('/')[:-1]
    save_dir = '/'.join(save_dir) + '/figs/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = torch.load(load_model)
    else:
        model = torch.load(load_model, map_location='cpu')

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()
    print('Start Evaluation ...')
    fig_id = 0
    for img, mask, regr, heatmap, dropmasks in tqdm(test_dataset):
        fig, axes = plt.subplots(5, 1, figsize=(16, 16))
        img2show = img.data.cpu().numpy()
        img = img.unsqueeze(0)
        axes[0].set_title('Input image')
        axes[0].imshow(np.rollaxis(img2show, 0, 3))

        axes[1].set_title('Ground truth mask')
        axes[1].imshow(mask)
        with torch.no_grad():
            output = model(img.to(device))

        if type(output) is list:
            output = output[-1]
            output = output['hm'] if type(output) is dict else output
        logits = output[0, 0].data.cpu().numpy()
        axes[2].set_title('Model predictions')
        axes[2].imshow(logits)

        axes[3].set_title('Model predictions thresholded')
        axes[3].imshow(logits > args.threshold)

        axes[4].set_title('Ground Truth Gaussian Kernel')
        axes[4].imshow(heatmap)
        plt.tight_layout()
        plt.savefig(save_dir + '{}.png'.format(fig_id))
        fig_id += 1
        plt.close()


    test = pd.read_csv(PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv(save_dir + '/predictions.csv', index=False)
    test.head()

if __name__ == '__main__':
    main()