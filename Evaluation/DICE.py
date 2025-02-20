import os
import cv2
import numpy as np

def IoU_score(pred, target):
    intersection = (pred * target).sum()
    union = pred + target
    union[union > 0] = 1
    union_ = union.sum()
    IoU = intersection / union_
    return IoU

def dice_loss(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    # union_ = pred + target
    # union_[union_ > 0] = 1
    # union = union_.sum()
    union = pred.sum() + target.sum()
    dice_score = (2 * intersection + epsilon) / (union + epsilon)
    # dice_loss = 1 - dice_score
    return dice_score

def IoU_CVFDLO():
    img_list = os.listdir('../data/test_labels')
    IoU_dict = {}
    for img_name in img_list:
        pred = cv2.imread(os.path.join('../data/test_predicts_mask_R101/wo ISOPair', img_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join('../data/test_labels', img_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (640, 360))
        pred[pred < 31] = 0
        pred[pred != 0] = 1
        target[target < 10] = 0
        target[target != 0] = 1
        IoU = IoU_score(pred, target)
        IoU_dict[img_name] = IoU
        print(img_name + ':' + str(IoU))

    avg_IoU = np.mean([v for v in IoU_dict.values()])
    avg_IoU_C1 = np.mean([v for key, v in IoU_dict.items() if key[1] == '1'])
    avg_IoU_C2 = np.mean([v for key, v in IoU_dict.items() if key[1] == '2'])
    avg_IoU_C3 = np.mean([v for key, v in IoU_dict.items() if key[1] == '3'])
    print("avg_IoU:", avg_IoU)
    print("avg_IoU_C1:", avg_IoU_C1)
    print("avg_IoU_C2:", avg_IoU_C2)
    print("avg_IoU_C3:", avg_IoU_C3)

def dice_CVFDLO():
    img_list = os.listdir('../data/test_labels')
    dice_dict = {}
    for img_name in img_list:
        pred = cv2.imread(os.path.join('../data/test_predicts_label/wo CC', img_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join('../data/test_labels', img_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.resize(target, (640, 360))
        pred[pred < 31] = 0
        pred[pred != 0] = 1
        target[target < 10] = 0
        target[target != 0] = 1
        dice = dice_loss(pred, target)
        dice_dict[img_name] = dice
        print(img_name + ':' + str(dice))

    avg_dice = np.mean([v for v in dice_dict.values()])
    avg_dice_C1 = np.mean([v for key, v in dice_dict.items() if key[1] == '1'])
    avg_dice_C2 = np.mean([v for key, v in dice_dict.items() if key[1] == '2'])
    avg_dice_C3 = np.mean([v for key, v in dice_dict.items() if key[1] == '3'])
    print("avg_dice:", avg_dice)
    print("avg_dice_C1:", avg_dice_C1)
    print("avg_dice_C2:", avg_dice_C2)
    print("avg_dice_C3:", avg_dice_C3)

def dice_reconst():
    pred_dir = '../data/LAB_imgs_1028_DLO/reconst_compare/reconsts'
    mask_dir = '../data/LAB_imgs_1028_DLO/reconst_compare/masks'
    img_list = os.listdir(pred_dir)
    dice_dict = {}
    for img_name in img_list:
        pred = cv2.imread(os.path.join(pred_dir, img_name), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(os.path.join(mask_dir, img_name), cv2.IMREAD_GRAYSCALE)
        vis(pred, target, img_name)
        pred[pred < 10] = 0
        pred[pred != 0] = 1
        target[target < 10] = 0
        target[target != 0] = 1
        dice = dice_loss(pred, target)
        dice_dict[img_name] = dice
        print(img_name + ':' + str(dice))
    avg_dice = np.mean([v for v in dice_dict.values()])
    print("avg_dice:", avg_dice)

def vis(pred, target, name):
    colored_image1 = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
    colored_image2 = cv2.applyColorMap(target, cv2.COLORMAP_OCEAN)
    alpha = 0.5  # 透明度
    merged_image = cv2.addWeighted(colored_image1, alpha, colored_image2, 1 - alpha, 0)
    cv2.imwrite('reprojection_loss/{}'.format(name), merged_image)


if __name__ == "__main__":
    # dice_reconst()
    dice_CVFDLO()
    # IoU_CVFDLO()