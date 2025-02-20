import cv2
import imageio
from tqdm import tqdm
import time
import torch
import numpy as np
import os

img_list = os.listdir("../../TEMP_DATA/gt_imgs")
for img_fname in tqdm(img_list):
    img = cv2.imread(os.path.join("../../TEMP_DATA/gt_imgs", img_fname))
    np.save(os.path.join("../../TEMP_DATA/gt_npys", img_fname[:-4] + '.npy'), img)