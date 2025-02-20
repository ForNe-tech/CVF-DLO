import os
import cv2
import numpy as np

def dice_loss(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    # union_ = pred + target
    # union_[union_ > 0] = 1
    # union = union_.sum()
    union = pred.sum() + target.sum()
    dice_score = (2 * intersection + epsilon) / (union + epsilon)
    # dice_loss = 1 - dice_score
    return dice_score

if __name__ == "__main__":

    S_list = ['S1', 'S2', 'S3']
    dice_dict = {}
    for s in S_list:
        img_list = os.listdir('../dataset/{}/predicts'.format(s))
        for img_name in img_list:
            pred = cv2.imread(os.path.join('../dataset/{}/predicts_wCC'.format(s), img_name), cv2.IMREAD_GRAYSCALE)
            target = np.load(os.path.join('../dataset/{}/gt_labels'.format(s), img_name[:-4] + '.npy'))
            pred_h, pred_w = pred.shape[0], pred.shape[1]
            target_h, target_w = target.shape[0], target.shape[1]
            size_h = min(pred_h, target_h)
            size_w = min(pred_w, target_w)
            pred = pred[0:size_h, 0:size_w]
            target = target[0:size_h, 0:size_w]
            pred[pred < 10] = 0
            pred[pred != 0] = 1
            target[target != 0] = 1
            dice = dice_loss(pred, target)
            dice_dict['{}_'.format(s) + img_name] = dice
            print('{}_'.format(s) + img_name + ":" + str(dice))

    avg_dice = np.mean([v for v in dice_dict.values()])
    avg_dice_S1 = np.mean([v for key, v in dice_dict.items() if key[1] == '1'])
    avg_dice_S2 = np.mean([v for key, v in dice_dict.items() if key[1] == '2'])
    avg_dice_S3 = np.mean([v for key, v in dice_dict.items() if key[1] == '3'])
    print("avg_dice:", avg_dice)
    print("avg_dice_S1:", avg_dice_S1)
    print("avg_dice_S2:", avg_dice_S2)
    print("avg_dice_S3:", avg_dice_S3)