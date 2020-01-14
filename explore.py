import pandas as pd
import numpy as np
from utils import add_number_of_cars, remove_out_image_cars

import torch
import torch.utils.data
from models.centernet_models import create_model

heads = {'hm': 8}

model = create_model('res_34', heads, 0)

print(model)