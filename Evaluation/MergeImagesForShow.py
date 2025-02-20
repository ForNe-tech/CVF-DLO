import cv2
import os

img = cv2.imread('../data/test_imgs/c1_16.png')
pred = cv2.imread('../data/Ablation Study mask/test/c1_16.png')

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
pred = cv2.dilate(pred, kernel, 1000)

merge = img * 0.5 + pred * 0.5
cv2.imwrite('MergeImages.jpg', merge)

print(1)