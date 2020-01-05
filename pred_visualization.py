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
print('Loading ...')
PATH = './data/'
test = pd.read_csv(PATH + 'sample_submission.csv')
train = pd.read_csv(PATH + 'train.csv')  # .sample(n=20).reset_index()
train_images_dir = PATH + 'train_images/{}.jpg'
test_images_dir = PATH + 'test_images/{}.jpg'
df_test = test
test_dataset = CarDataset(train, train_images_dir, training=False)
load_model = '/scratch/bz1030/auto_drive/run/model_HG2_stack_2_features_256_MSE_2020-01-04_07-59-51/model_7.pth'
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


model.eval()
print('Start Evaluation ...')
fig_id = 0
for img, mask, regr, heatmap in tqdm(test_dataset):
    print(np.where(mask > 0))
    fig, axes = plt.subplots(5, 1, figsize=(16, 16))
    axes[0].set_title('Input image')
    axes[0].imshow(np.rollaxis(img, 0, 3))

    axes[1].set_title('Ground truth mask')
    axes[1].imshow(mask)

    output = model(torch.tensor(img[None]).to(device))
    if type(output) is list:
        output = output[-1]

    logits = output[0, 0].data.cpu().numpy()

    axes[2].set_title('Model predictions')
    axes[2].imshow(logits)

    axes[3].set_title('Model predictions thresholded')
    axes[3].imshow(logits > 0)

    axes[4].set_title('Ground Truth Gaussian Kernel')
    axes[4].imshow(heatmap)
    plt.tight_layout()
    plt.savefig(save_dir + '{}.png'.format(fig_id))
    fig_id += 1


test = pd.read_csv(PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv(save_dir + '/predictions.csv', index=False)
test.head()