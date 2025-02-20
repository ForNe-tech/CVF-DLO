from CVF.segment.predict import SegNet
import os
import cv2
from tqdm import tqdm


ckpt_seg_name = "checkpoints/EWD/CP_segmentation_resnet50.pth"

network_seg = SegNet(model_name="deeplabv3plus_resnet50", checkpoint_path=ckpt_seg_name, img_w=640, img_h=360)

img_list = os.listdir("data/test_imgs")

os.makedirs("data/test_masks_r50", exist_ok=True)

for img_name in tqdm(img_list):

    source_img = cv2.imread(os.path.join("data/test_imgs", img_name), cv2.IMREAD_UNCHANGED)
    mask_img = network_seg.predict_img(source_img)

    cv2.imwrite(os.path.join("data/test_masks_r50", img_name), mask_img)