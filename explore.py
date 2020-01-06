import pandas as pd
import numpy as np
from utils import add_number_of_cars, remove_out_image_cars


df = pd.read_csv('./data/train_fixed.csv')
df = remove_out_image_cars(df)
df = add_number_of_cars(df)
print(df.numcars.mean(), df.numcars.sum())