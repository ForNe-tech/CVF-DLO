import os, cv2
from CVF.core import Pipeline
import numpy as np
import arrow
import matplotlib

if __name__ == "__main__":

    img_list = os.listdir("data/test_imgs")

    total_time = {}
    seg_time = {}
    proc_time = {}

    IMG_W = 640
    IMG_H = 360

    ckpt_seg_name = "checkpoints/EWD/CP_segmentation_resnet50.pth"

    cmap = matplotlib.cm.get_cmap('Set1', 10)
    colors = cmap(np.linspace(0, 1, 10))
    rgb_colors_1 = (colors[:, :3] * 255).astype(int)
    cmap = matplotlib.cm.get_cmap('Set2', 10)
    colors = cmap(np.linspace(0, 1, 10))
    rgb_colors_2 = (colors[:, :3] * 255).astype(int)
    rgb_colors = np.vstack((rgb_colors_1, rgb_colors_2))

    p = Pipeline(checkpoint_seg=ckpt_seg_name, colors=rgb_colors, img_w=IMG_W, img_h=IMG_H, if_debug=False, scene='EWD')

    for img_name in img_list:

        # img_name = 'c1_39.png'
        print(img_name)

        IMG_PATH = "data/test_imgs/{}".format(img_name)
        MASK_PATH = "data/test_masks/{}".format(img_name)

        source_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        source_img = cv2.resize(source_img, (IMG_W, IMG_H))
        mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (IMG_W, IMG_H))

        t0 = arrow.utcnow()

        img_out, _, skel_m, times, _, _, _ = p.run(source_img=source_img, mask_img=None, mask_th=31, verbose=True)

        total_time[img_name] = (arrow.utcnow() - t0).total_seconds() * 1000
        print('Out Total time: {:.5f}'.format((arrow.utcnow() - t0).total_seconds()))

        out_dir = 'data/test_predicts_mask_R50/test5'
        os.makedirs(out_dir, exist_ok=True)

        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

        cv2.imwrite(out_dir + '/{}'.format(img_name), img_out)

    del total_time['c1_0.png']
    avg_time = np.mean([v for v in total_time.values()])
    avg_time_C1 = np.mean([v for key, v in total_time.items() if key[1] == '1'])
    avg_time_C2 = np.mean([v for key, v in total_time.items() if key[1] == '2'])
    avg_time_C3 = np.mean([v for key, v in total_time.items() if key[1] == '3'])
    print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    print("avg_time_C1:", avg_time_C1, "FPS:", 1000 / avg_time_C1)
    print("avg_time_C2:", avg_time_C2, "FPS:", 1000 / avg_time_C2)
    print("avg_time_C3:", avg_time_C3, "FPS:", 1000 / avg_time_C3)

    # del seg_time['c1_0.png']
    # avg_time = np.mean([v for v in seg_time.values()])
    # avg_time_C1 = np.mean([v for key, v in seg_time.items() if key[1] == '1'])
    # avg_time_C2 = np.mean([v for key, v in seg_time.items() if key[1] == '2'])
    # avg_time_C3 = np.mean([v for key, v in seg_time.items() if key[1] == '3'])
    # print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    # print("avg_time_C1:", avg_time_C1, "FPS:", 1000 / avg_time_C1)
    # print("avg_time_C2:", avg_time_C2, "FPS:", 1000 / avg_time_C2)
    # print("avg_time_C3:", avg_time_C3, "FPS:", 1000 / avg_time_C3)
    #
    # del proc_time['c1_0.png']
    # avg_time = np.mean([v for v in proc_time.values()])
    # avg_time_C1 = np.mean([v for key, v in proc_time.items() if key[1] == '1'])
    # avg_time_C2 = np.mean([v for key, v in proc_time.items() if key[1] == '2'])
    # avg_time_C3 = np.mean([v for key, v in proc_time.items() if key[1] == '3'])
    # print("avg_time:", avg_time, "FPS:", 1000 / avg_time)
    # print("avg_time_C1:", avg_time_C1, "FPS:", 1000 / avg_time_C1)
    # print("avg_time_C2:", avg_time_C2, "FPS:", 1000 / avg_time_C2)
    # print("avg_time_C3:", avg_time_C3, "FPS:", 1000 / avg_time_C3)