import os

import cv2
import numpy as np

from CVF.core import Pipeline

import arrow
from time import time

def detect_pink_green_teal(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Pink
    lower_pink = np.array([0, 20, 75])
    upper_pink = np.array([20, 255, 255])
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    lower_pink = np.array([160, 20, 75])
    upper_pink = np.array([180, 255, 255])
    mask_pink += cv2.inRange(hsv, lower_pink, upper_pink)

    # Green
    lower_green = np.array([25, 20, 75])
    upper_green = np.array([50, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Teal
    lower_teal = np.array([50, 20, 20])
    upper_teal = np.array([120, 255, 255])
    mask_teal = cv2.inRange(hsv, lower_teal, upper_teal)

    # Combine masks
    mask = mask_pink + mask_green + mask_teal
    mask[mask != 0] = 255

    # Denoise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

if __name__ == "__main__":

    set_list = ['S1', 'S2', 'S3']

    total_time = {}

    Plotting_time = []

    IMG_W = 672
    IMG_H = 896

    TH = 1

    colors = [[0, 255, 0], [0, 0, 255], [255, 0, 0],
               [0, 255, 255], [255, 255, 0], [255, 0, 255],
               [0, 127, 0], [0, 0, 127], [127, 0, 0],
               [0, 127, 127], [127, 127, 0], [127, 0, 127],
               [0, 64, 0], [0, 0, 64], [64, 0, 0],
               [0, 64, 64], [64, 64, 0], [64, 0, 64]]
    rgb_colors = np.array(colors)

    p = Pipeline(checkpoint_seg=None, colors=rgb_colors, img_w=IMG_W, img_h=IMG_H, if_debug=False, scene='SBHC')

    for set in set_list:

        img_list = os.listdir('dataset/{}/images'.format(set))

        os.makedirs('dataset/{}/predicts_test4'.format(set), exist_ok=True)


        for img_name in img_list:
            # img_name = 'img24.jpg'
            print(set + '_' + img_name)
            source_img = cv2.imread('dataset/{}/images/{}'.format(set, img_name))

            t0 = arrow.utcnow()

            mask_img = detect_pink_green_teal(source_img)

            print('Seg time: {:5f}'.format((arrow.utcnow() - t0).total_seconds()))

            img_out, _, _, times, _, _, _ = p.run(source_img=source_img, mask_img=mask_img, mask_th=TH, verbose=True)
            img_out[mask_img[0:IMG_H, 0:IMG_W] < 31] = (0, 0, 0)

            total_time['{}_'.format(set) + img_name] = (arrow.utcnow() - t0).total_seconds() * 1000
            print('Out Total time: {:.5f}'.format((arrow.utcnow() - t0).total_seconds()))
            Plotting_time.append(times[-1])

            img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

            cv2.imwrite('dataset/{}/predicts_test4/{}'.format(set, img_name[:-4] + '.png'), img_out)

    avg_time = np.mean([v for v in total_time.values()])
    avg_time_S1 = np.mean([v for key, v in total_time.items() if key[1] == '1'])
    avg_time_S2 = np.mean([v for key, v in total_time.items() if key[1] == '2'])
    avg_time_S3 = np.mean([v for key, v in total_time.items() if key[1] == '3'])
    print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    print("avg_time_S1:", avg_time_S1, "FPS:", 1000 / avg_time_S1)
    print("avg_time_S2:", avg_time_S2, "FPS:", 1000 / avg_time_S2)
    print("avg_time_S3:", avg_time_S3, "FPS:", 1000 / avg_time_S3)

    print("avg_Plotting_time:", np.mean(Plotting_time))