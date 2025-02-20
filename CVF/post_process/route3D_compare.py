import numpy as np
import matplotlib.pyplot as plt
import os
import json

def vis_cables(cable1, cable2, label, visdir):
    path1 = np.array(cable1)
    path2 = np.array(cable2)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(path1[:, 0], path1[:, 1], path1[:, 2], zorder=1, c='r', s=1)
    ax.scatter(path2[:, 0], path2[:, 1], path2[:, 2], zorder=1, c='b', s=1)
    plt.savefig(os.path.join(visdir, '{}_compare.png'.format(label)))

if __name__ == "__main__":
    base_dir = '../../data/LAB_imgs_1028_DLO/LAB_CABIN_Cables/route3D'
    cable_est_dir = '../../data/LAB_imgs_1028_DLO/route3D_select_bs'
    cable_gt_dir = '../../data/LAB_imgs_1028_DLO/LAB_CABIN_Cables/route3D_bs'
    vis_dir = '../../data/LAB_imgs_1028_DLO/route3D_compare'
    cable_list = os.listdir(base_dir)
    for cable_name in cable_list:
        cable_name = cable_name[:-5]
        cable_est_path = os.path.join(cable_est_dir, '{}_bs.json'.format(cable_name))
        cable_gt_path = os.path.join(cable_gt_dir, '{}_bs.json'.format(cable_name))
        with open(cable_est_path, 'r', encoding='utf-8') as f1:
            cable_est = json.load(f1)
        with open(cable_gt_path, 'r', encoding='utf-8') as f2:
            cable_gt = json.load(f2)
        vis_cables(cable_est, cable_gt, cable_name, vis_dir)
        print(1)