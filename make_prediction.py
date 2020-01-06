import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm#_notebook as tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import coords2str, extract_coords, add_number_of_cars
from train import CarDataset
import argparse
import time

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
    test_images_dir = PATH + 'test_images/{}.jpg'
    df_test = test
    test_dataset = CarDataset(df_test, test_images_dir, sigma=1, training=False)
    load_model = args.load_model
    predictions = []
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
    for img, _, _, _ in tqdm(test_loader):
        with torch.no_grad():
            output = model(img.to(device))
        if type(output) is list:
            output = output[-1]
        output = output.data.cpu().numpy()
        for out in output:
            coords = extract_coords(out, args.threshold)
            s = coords2str(coords)
            predictions.append(s)

    save_dir = load_model.split('/')[:-1]
    save_dir = '/'.join(save_dir)
    test = pd.read_csv(PATH + 'sample_submission.csv')
    test['PredictionString'] = predictions
    test.to_csv(save_dir + '/predictions_{}.csv'.format(args.threshold), index=False)
    test = add_number_of_cars(test)
    avg_cars, sum_cars = test.numcars.mean(), test.numcars.sum()
    with open(save_dir +'/stats_{}.txt'.format(args.threshold), 'a+') as f:
        f.write('Average:'+ str(avg_cars) + '\n')
        f.write('Total:'+ str(avg_cars) + '\n')




if __name__ == '__main__':
    main()
