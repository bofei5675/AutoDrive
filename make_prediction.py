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
load_model = '/scratch/bz1030/auto_drive/saved_bilinear/model_19.pth'
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
    output = output.data.cpu().numpy()
    for out in output:
        coords = extract_coords(out)
        s = coords2str(coords)
        predictions.append(s)
save_dir = load_model.split('/')[:-1]
save_dir = '/'.join(load_model)
test = pd.read_csv(PATH + 'sample_submission.csv')
test['PredictionString'] = predictions
test.to_csv(save_dir + '/predictions.csv', index=False)
test.head()