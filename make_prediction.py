import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm#_notebook as tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import coords2str, extract_coords, add_number_of_cars, save_submission_file,\
    IMG_WIDTH, IMG_HEIGHT, MODEL_SCALE
import cv2
from train import CarDataset
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-lm', '--load-model', type=str, dest='load_model', default=None)
    args.add_argument('-t', '--threshold', type=float, dest='threshold', default=0)
    return args.parse_args()


def main():
    args = parse_args()
    print('Loading ...')
    PATH = './data/'
    test = pd.read_csv(PATH + 'sample_submission.csv')
    #test = test.iloc[:50]
    test_images_dir = PATH + 'test_images/{}.jpg'
    df_test = test
    test_dataset = CarDataset(df_test, test_images_dir, sigma=1, training=False)
    load_model = args.load_model
    predictions, predictions_dropmask = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model = torch.load(load_model)
    else:
        model = torch.load(load_model, map_location='cpu')

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)

    model.eval()
    print('Start Evaluation ...')
    for img, _, _, _, dropmasks in tqdm(test_loader):
        img = img.float().to(device)
        dropmasks = dropmasks.data.cpu().numpy()
        with torch.no_grad():
            output = model(img)
        if type(output) is list:
            output = output[-1]
            output = output['hm'] if type(output) is dict else output
        output = output.data.cpu().numpy()
        for out, test_mask in zip(output, dropmasks):
            # get unprocessed value
            coords = extract_coords(out, args.threshold)
            s = coords2str(coords)
            predictions.append(s)
            #test_mask = cv2.resize(test_mask[0], (IMG_WIDTH // MODEL_SCALE,  IMG_HEIGHT // MODEL_SCALE))
            # test_mask = np.where(test_mask > 255 // 2, 100, 0)  # subtract from logits
            #print(test_mask.shape, out[0].shape)
            #print(out[0].mean())
            #print(sth.sum())
            #out[0, test_mask > (255 // 2)] = -100
            #print(out[0].mean())
            #coords = extract_coords(out, args.threshold)
            #s = coords2str(coords)
            #predictions_dropmask.append(s)

    save_dir = load_model.split('/')[:-1]
    save_dir = '/'.join(save_dir)
    save_submission_file(test.copy(), save_dir, predictions, args.threshold, 'origin')
    #save_submission_file(test.copy(), save_dir, predictions_dropmask, args.threshold, 'drop')


if __name__ == '__main__':
    main()
