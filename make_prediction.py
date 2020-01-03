import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm#_notebook as tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from utils import coords2str, CarDataset, MyUNet, double_conv, up, extract_coords

print('Loading ...')
PATH = './data/'
test = pd.read_csv(PATH + 'sample_submission.csv')
test_images_dir = PATH + 'test_images/{}.jpg'
df_test = test
test_dataset = CarDataset(df_test, test_images_dir, training=False)
load_model = '/scratch/bz1030/auto_drive/run/model_HG2_stack_2_features_256_2020-01-03_01-21-51/model_10.pth'
predictions = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    model = torch.load(load_model)
else:
    model = torch.load(load_model, map_location='cpu')


test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)

model.eval()
print('Start Evaluation ...')
for img, _, _ in tqdm(test_loader):
    with torch.no_grad():
        output = model(img.to(device))
    if type(output) is list:
        output = output[-1]
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out)
        s = coords2str(coords)
        predictions.append(s)
save_dir = load_model.split('/')[:-1]
save_dir = '/'.join(save_dir)
test = pd.read_csv(PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv(save_dir + '/predictions.csv', index=False)
test.head()