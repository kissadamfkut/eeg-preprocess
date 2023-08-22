import cv2
import numpy as np


def mask(img_c, img, lower_val, method):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# =============================================================================
#     lower_val = np.array([105,0,0])
# =============================================================================
    upper_val = np.array([255,255,255])
    mask = cv2.inRange(hsv, lower_val, upper_val)
    #cv2.imshow("inrange", mask)
    #cv2.waitKey(0)
    ranged_img = cv2.bitwise_and(img, img, mask = mask)
    background = np.zeros(img.shape, img.dtype)
    background = cv2.bitwise_not(background)
    mask_inv = cv2.bitwise_not(mask)
    #cv2.imshow("inrange_inv", mask_inv)
    #cv2.waitKey(0)
    masked_bg = cv2.bitwise_and(background, background, mask = mask_inv)
    if method == 'add':
        masked_img = cv2.add(ranged_img, masked_bg)
    if method == 'subtract':
        masked_img = cv2.subtract(img_c, masked_bg)
    return masked_img
    
