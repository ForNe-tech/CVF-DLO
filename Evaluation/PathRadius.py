import cv2
import os
from skimage.morphology import skeletonize
import numpy as np

def remove_small_connected_components(binary, min_area):

    # 使用connectedComponentsWithStats找到所有联通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # 遍历所有联通区域
    for i in range(1, num_labels):  # 0是背景标签，所以从1开始
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            labels[labels == i] = 0  # 将小联通区域设置为背景

    # 根据labels矩阵重新创建二值图像
    new_image = np.zeros_like(binary)
    new_image[labels > 0] = 255

    return new_image


if __name__ == "__main__":
    out_dir = '../data/LAB_imgs_1028_DLO/reconst_compare/reconsts'
    mask_dir = '../data/LAB_imgs_1028_DLO/route3D_reconst_only_ORIGIN'
    mask_files = os.listdir(mask_dir)
    for mask_file in mask_files:
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = ~mask
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        # cv2.imshow('mask', mask)
        # cv2.waitKey(0)

        mask = remove_small_connected_components(mask, 100)

        skeleton = skeletonize(mask, method='lee')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        dilated = cv2.dilate(skeleton, kernel, iterations=1)

        # translation_matrix = np.float32([[1, 0, 0], [0, 1, 15]])
        # translated_image = cv2.warpAffine(dilated, translation_matrix, (896, 504))

        dilated = dilated[15:504, 0:896]

        cv2.imwrite(os.path.join(out_dir, mask_file[:-4] + '.png'), dilated)