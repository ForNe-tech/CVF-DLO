import os
from tqdm import tqdm
import cv2
import numpy as np


def get_instance_nums(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    instance_colors, counts = np.unique(img[np.nonzero(img)], return_counts=True)
    instance_imgs = {}
    for i, instance_color in enumerate(instance_colors):
        new_img = np.zeros_like(img)
        new_img[img == instance_color] = 1
        instance_imgs[instance_color] = new_img
    return len(instance_colors), instance_imgs


def getUnion(pred, target):
    pred_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    pred_g[pred_g > 0] = 1
    target_g[target_g > 0] = 1
    union_g = pred_g + target_g
    union_g[union_g > 0] = 1
    return union_g.sum()


def PQ(pred_c, target_c, epsilon=1e-6):
    pred = cv2.cvtColor(pred_c, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target_c, cv2.COLOR_BGR2GRAY)
    pred[pred > 0] = 1
    target[target > 0] = 1
    intersection = (pred * target).sum()
    union_ = pred + target
    union_[union_ > 0] = 1
    union = union_.sum()
    IoU_score = (intersection + epsilon) / (union + epsilon)
    return intersection * IoU_score, union


if __name__ == "__main__":

    PQ_dict = {}

    img_list = os.listdir('../LABD/BWH/imgs')

    pred_dir = '../LABD/BWH/predict_labels_r101'
    target_dir = '../LABD/BWH/labels'

    pred_files = os.listdir(pred_dir)
    target_files = os.listdir(target_dir)

    for img_name in img_list:

        prefix = img_name[:-4] + '_'

        pred_dict = {}
        target_dict = {}

        # 删选符合条件的文件名
        matching_pred_files = [file for file in pred_files if file.startswith(prefix)]

        for matching_pred_file in matching_pred_files:
            index = int(matching_pred_file[len(prefix):-4])
            pred = cv2.imread(os.path.join(pred_dir, matching_pred_file), cv2.IMREAD_UNCHANGED)
            pred_dict[index] = pred

        # 删选符合条件的文件名
        matching_target_files = [file for file in target_files if file.startswith(prefix)]

        for matching_target_file in matching_target_files:
            index = int(matching_target_file[len(prefix):-4])
            target = cv2.imread(os.path.join(target_dir, matching_target_file), cv2.IMREAD_UNCHANGED)
            target_dict[index] = target

        PQ_sum = 0
        pixel_sum = 0

        for target_idx, target in target_dict.items():
            PQ_score_possible = []
            for pred_idx, pred in pred_dict.items():
                PQ_score_possible.append(PQ(pred, target))
            sorted_PQ_score_possible = sorted(PQ_score_possible, key=lambda x:x[0], reverse=True)
            PQ_score = sorted_PQ_score_possible[0]
            PQ_sum += PQ_score[0]
            pixel_sum += PQ_score[1]

        print(img_name + ' PQ_score:', PQ_sum/ pixel_sum)

        PQ_dict[img_name] = PQ_sum / pixel_sum

    avg_PQ = np.mean([v for v in PQ_dict.values()])
    print("avg_PQ:", avg_PQ)