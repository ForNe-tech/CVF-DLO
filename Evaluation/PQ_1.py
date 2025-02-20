import os
from tqdm import tqdm
import cv2
import numpy as np
import time


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

def get_instance_nums_2(img):
    print(img.shape)
    if len(img.shape)==3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    instance_colors, counts = np.unique(img[np.nonzero(img)], return_counts=True)
    instance_imgs = {}
    for i, instance_color in enumerate(instance_colors):
        new_img = np.zeros_like(img)
        new_img[img == instance_color] = 1
        instance_imgs[instance_color] = new_img
    return len(instance_colors), instance_imgs

def PQ(pred, target, epsilon=1e-6):
    intersection_ = pred * target
    intersection = intersection_.sum()
    union_ = pred + target
    union_[union_ > 0] = 1
    union = union_.sum()
    IoU_score = (intersection + epsilon) / (union + epsilon)
    merge = intersection_ * 127 + union_ * 127
    # if intersection > 100 and union > 100:
    #     cv2.imwrite('intersection_union_{}_{}_{}_{}.png'.format(pred.sum(), target.sum(), intersection, union), merge)
    return intersection * IoU_score

def getUnion(pred, target):
    pred_g = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    if len(target.shape) == 3:
        target_g = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    else:
        target_g = target.copy()
    pred_g[pred_g > 0] = 1
    target_g[target_g > 0] = 1
    union_g = pred_g + target_g
    union_g[union_g > 0] = 1
    return union_g.sum(), pred_g.sum()


if __name__ == "__main__":

    s = 'S2'

    img_list = os.listdir('../dataset/{}/images'.format(s))

    img_name = 'img73.png'

    pred = cv2.imread(os.path.join('../dataset/{}/predicts_test'.format(s), img_name), cv2.IMREAD_UNCHANGED)
    # target = cv2.imread(os.path.join('../data/test_labels', img_name), cv2.IMREAD_UNCHANGED)
    target = np.load(os.path.join('../dataset/{}/gt_labels'.format(s), img_name[:-4] + '.npy'))
    # target = cv2.imread(os.path.join('../dataset/{}/mBEST_predicts'.format(s), img_name), cv2.IMREAD_UNCHANGED)
    pred_h, pred_w = pred.shape[0], pred.shape[1]
    target_h, target_w = target.shape[0], target.shape[1]
    size_h = min(pred_h, target_h)
    size_w = min(pred_w, target_w)
    pred = pred[0:size_h, 0:size_w]
    target = target[0:size_h, 0:size_w]
    
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    pred_gray[pred_gray < 31] = 0
    pred_gray[pred_gray != 0] = 1
    union_show = np.zeros_like(pred)
    try:
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        target_gray[target_gray < 31] = 0
    except:
        target_gray = target.copy()
        target_gray[target_gray < 1] = 0
    target_gray[target_gray != 0] = 1
    union = pred_gray + target_gray
    union_show[pred_gray == 1] = (255, 0, 0)   # 蓝色
    union_show[target_gray == 1] = (0, 255, 0)      # 绿色
    union_show[union == 2] = (0, 0, 255)       # 红色
    cv2.imwrite('show_results/pred_target_show.jpg', union_show)

    pred_instance_num, pred_instances = get_instance_nums_2(pred)
    target_instance_num, target_instances = get_instance_nums_2(target)
    # for pred in pred_instances.values():
    #     cv2.imwrite('pred.png', pred * 255)

    pixel_N, pixel_pred = getUnion(pred, target)

    PQ_sum = []

    for pred in pred_instances.values():
        PQ_score_possible = []
        for target in target_instances.values():
            PQ_score_possible.append(PQ(pred, target))
        PQ_score = max(PQ_score_possible)
        PQ_sum.append(PQ_score)

    print(PQ_sum, pixel_N, pixel_pred)
    print(img_name + ' PQ_score:', sum(PQ_sum) / pixel_N)