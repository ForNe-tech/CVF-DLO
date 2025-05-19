import os

import cv2
import numpy as np

from CVF3D.core import Pipeline

import arrow
from time import time

import matplotlib

if __name__ == "__main__":

    total_time = {}

    Plotting_time = []

    IMG_W = 896
    IMG_H = 504

    TH = 31

    cmap = matplotlib.cm.get_cmap('Set1', 10)
    colors = cmap(np.linspace(0, 1, 10))
    rgb_colors_1 = (colors[:, :3] * 255).astype(int)
    cmap = matplotlib.cm.get_cmap('Set2', 10)
    colors = cmap(np.linspace(0, 1, 10))
    rgb_colors_2 = (colors[:, :3] * 255).astype(int)
    rgb_colors = np.vstack((rgb_colors_1, rgb_colors_2))

    ckpt_seg_name = "checkpoints/EWD/CP_segmentation.pth"

    p = Pipeline(checkpoint_seg=ckpt_seg_name, colors=rgb_colors, img_w=IMG_W, img_h=IMG_H, if_debug=False, scene='BWH')

    dir = 'DATASETS/BWH'

    img_dir = f'{dir}/imgs'
    mask_dir = f'{dir}/masks'

    img_list = os.listdir(img_dir)

    mask_out_dir = f'{dir}/predict_masks'
    label_out_dir = f'{dir}/predict_labels'

    os.makedirs(mask_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    for img_name in img_list:
        # img_name = 'temp12.png'
        print(img_name)
        source_img = cv2.imread(img_dir + f'/{img_name}', cv2.IMREAD_UNCHANGED)

        t0 = arrow.utcnow()

        mask_name = img_name[:-4] + '.png'
        mask_img = cv2.imread(mask_dir + f'/{mask_name}', cv2.IMREAD_UNCHANGED)

        print('Seg time: {:5f}'.format((arrow.utcnow() - t0).total_seconds()))

        img_out, _, route_list, times, _, _, _ = p.run(source_img=source_img, mask_img=mask_img, mask_th=TH, verbose=True)
        # img_out[mask_img[0:IMG_H, 0:IMG_W] < 31] = (0, 0, 0)

        total_time[img_name] = (arrow.utcnow() - t0).total_seconds() * 1000
        print('Out Total time: {:.5f}'.format((arrow.utcnow() - t0).total_seconds()))
        Plotting_time.append(times[-1])

        # img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
        #
        # cv2.imwrite(mask_out_dir + f'/{mask_name}', img_out)
        #
        # for i, route_img in enumerate(route_list):
        #     route_name = img_name[:-4] + f'_{i}.png'
        #     route_img = cv2.cvtColor(route_img, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite(label_out_dir + f'/{route_name}', route_img)
    del total_time['b2_0.jpg']
    avg_time = np.mean([v for v in total_time.values()])
    print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    print("avg_Plotting_time:", np.mean(Plotting_time))