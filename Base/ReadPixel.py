import cv2

#鼠标点击响应事件
def get_bgr_value(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("RGB value is",img[y,x]/255)

def get_gray_value(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        gray_value = gray[y, x]
        print("Gray value is", gray_value)

def get_hsv_value(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print("HSV value is",HSV[y,x]/255)

# 直接读为灰度图像
img = cv2.imread('../PicReference/caa671a1b8714ccc744104bafce8a49.jpg', cv2.IMREAD_UNCHANGED)

# 将图像缩放到scale x scale
height, width = img.shape[:2]
if height > width:
    scale = 700 / height
else:
    scale = 700 / width
img = cv2.resize(img, None, fx=scale, fy=scale)

#BGR转化为gray、HSV
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
HSV = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#

cv2.imshow('image_RGB',img)
# cv2.imshow('image_GRAY',gray)
# cv2.imshow("image_HSV",HSV)
cv2.setMouseCallback("image_RGB",get_bgr_value)
# cv2.setMouseCallback("image_GRAY",get_gray_value)
# cv2.setMouseCallback("image_HSV",get_hsv_value)
cv2.waitKey(0)
