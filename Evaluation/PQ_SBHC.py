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
    target_g = target
    pred_g[pred_g > 0] = 1
    target_g[target_g > 0] = 1
    union_g = pred_g + target_g
    union_g[union_g > 0] = 1
    return union_g.sum(), pred_g.sum()


def PQ(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    union_ = pred + target
    union_[union_ > 0] = 1
    union = union_.sum()
    IoU_score = (intersection + epsilon) / (union + epsilon)
    return intersection * IoU_score


if __name__ == "__main__":

    set_list = ['S1', 'S2', 'S3']
    PQ_dict = {}
    PP_dict = {}

    for set in set_list:

        img_list = os.listdir('../SBHC/{}/images'.format(set))

        for img_name in img_list:

            pred = cv2.imread(os.path.join('../SBHC/{}/predicts_wCC'.format(set), img_name[:-4] + '.png'), cv2.IMREAD_UNCHANGED)
            # target = cv2.imread(os.path.join('../SBHC/{}/gt_images'.format(set), img_name[:-4] + '.png'), cv2.IMREAD_UNCHANGED)
            target = np.load(os.path.join('../SBHC/{}/gt_labels'.format(set), img_name[:-4] + '.npy'))
            pred_h, pred_w = pred.shape[0], pred.shape[1]
            target_h, target_w = target.shape[0], target.shape[1]
            size_h = min(pred_h, target_h)
            size_w = min(pred_w, target_w)
            pred = pred[0:size_h, 0:size_w]
            target = target[0:size_h, 0:size_w]

            pred_instance_num, pred_instances = get_instance_nums(pred)
            target_instance_num, target_instances = get_instance_nums(target)
            # for pred in pred_instances.values():
            #     cv2.imwrite('pred.png', pred * 255)

            pixel_N, pixel_pred = getUnion(pred, target)
            # for target in target_instances.values():
            #     pixel_N += target.sum()

            PQ_sum = 0

            for pred in pred_instances.values():
                PQ_score_possible = []
                for target in target_instances.values():
                    PQ_score_possible.append(PQ(pred, target))
                PQ_score = max(PQ_score_possible)
                PQ_sum += PQ_score

            print(set + '_' + img_name + ' PQ_score:', PQ_sum / pixel_N, 'pred_instance:', pred_instance_num, 'target_instance:',
                  target_instance_num)

            PQ_dict[set + '_' + img_name] = PQ_sum / pixel_N
            PP_dict[set + '_' + img_name] = pixel_pred

    avg_PQ = np.mean([v for v in PQ_dict.values()])
    avg_PQ_S1 = np.mean([v for key, v in PQ_dict.items() if key[1] == '1'])
    avg_PQ_S2 = np.mean([v for key, v in PQ_dict.items() if key[1] == '2'])
    avg_PQ_S3 = np.mean([v for key, v in PQ_dict.items() if key[1] == '3'])
    print("avg_PQ:", avg_PQ)
    print("avg_PQ_S1:", avg_PQ_S1)
    print("avg_PQ_S2:", avg_PQ_S2)
    print("avg_PQ_S3:", avg_PQ_S3)

    avg_PP = np.mean([v for v in PP_dict.values()])
    avg_PP_S1 = np.mean([v for key, v in PP_dict.items() if key[1] == '1'])
    avg_PP_S2 = np.mean([v for key, v in PP_dict.items() if key[1] == '2'])
    avg_PP_S3 = np.mean([v for key, v in PP_dict.items() if key[1] == '3'])
    print("avg_PP:", avg_PP)
    print("avg_PP_S1:", avg_PP_S1)
    print("avg_PP_S2:", avg_PP_S2)
    print("avg_PP_S3:", avg_PP_S3)