import os, cv2
from CVF3D.core import Pipeline
import json
import numpy as np
import matplotlib.pyplot as plt

def draw_ends_in_masks(img, img_name, ends_dict):
    plt.figure(figsize=(32, 18))
    plt.imshow(img)
    for end_key, end_value in ends_dict.items():
        y, x = end_value['point']
        y -= 5
        x -= 5
        plt.plot(x, y, 'ro')
        plt.text(x, y, str(end_key), fontsize=12, ha='right', color='black')
    plt.axis('off')
    plt.savefig(f'data/LAB_imgs_design_DLO/point_vis/{img_name}')
    plt.clf()



if __name__ == "__main__":

    DIR = "data/LAB_imgs_design_DLO"
    image_DIR = os.path.join(DIR, 'imgs')
    mask_DIR = os.path.join(DIR, 'masks_manual')
    route_DIR = os.path.join(DIR, 'route')
    seg_DIR = os.path.join(DIR, 'path2D')
    os.makedirs(route_DIR, exist_ok=True)
    os.makedirs(seg_DIR, exist_ok=True)

    image_list = os.listdir(image_DIR)

    IMG_W = 896
    IMG_H = 504

    p = Pipeline(img_w=IMG_W, img_h=IMG_H, if_debug=False, scene='BWH')

    for image_name in image_list:

        # image_name = 'temp15.png'

        print(image_name)

        IMG_PATH = os.path.join(image_DIR, image_name)
        MASK_PATH = os.path.join(mask_DIR, image_name)

        if not os.path.exists(MASK_PATH):
            continue

        source_img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

        mask_img = cv2.imread(MASK_PATH, cv2.IMREAD_UNCHANGED)

        img_out, _, route_seg_list, _, routes, end_pairs, ends_dict = p.run(source_img=source_img, mask_img=mask_img, mask_th=63)

        cv2.imwrite(os.path.join(route_DIR, image_name), img_out)
        #
        # for k, route_seg in enumerate(route_seg_list):
        #     cv2.imwrite(os.path.join(seg_DIR, (image_name[:-4] + '_{}.jpg').format(k)), route_seg)


        paths = {}
        for route_key, route_value in routes.items():
            paths[route_key] = (route_value['route'] - 13).tolist()
        ends = {}
        for end_key, end_value in ends_dict.items():
            ends[end_key] = {
                "type": end_value['point_type'],
                "border": end_value['border'],
                "route_label": int(end_value['route_label']),
                "route_end": end_value['route_end'],
                "pair_ends": end_value['pair_ends']
            }

        draw_ends_in_masks(source_img, image_name, ends_dict)

        with open(os.path.join(seg_DIR, image_name[:-4] + '_paths2D.json'), 'w', encoding='utf-8') as file1:
            json_string_1 = json.dumps(paths, ensure_ascii=True)
            file1.write(json_string_1)
            file1.close()

        with open(os.path.join(seg_DIR, image_name[:-4] + '_ends2D.json'), 'w', encoding='utf-8') as file2:
            json_string_2 = json.dumps(ends, ensure_ascii=True)
            file2.write(json_string_2)
            file2.close()

        print(1)