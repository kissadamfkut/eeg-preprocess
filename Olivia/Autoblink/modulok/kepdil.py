import cv2
import numpy as np
def picdil(n, stencil, img):
    kernel = np.ones((25,12), np.uint8)
# =============================================================================
#     kernel2 = np.ones((0,1), np.uint8)
#     img_dilated = cv2.erode(img.copy(), kernel, iterations = 1)
#     img_dilated = cv2.dilate(img.copy(), kernel2, iterations = 1)
# =============================================================================
    kernel2 = np.ones((0,1), np.uint8)
    img_dilated = cv2.erode(img.copy(),kernel2, iterations = 1)
    img_dilated = cv2.bitwise_not(img_dilated)
    
    res = cv2.bitwise_and(img_dilated, img_dilated, mask = stencil)
    #cv2.imshow("mask", res)
    #cv2.waitKey(0)
    
    h,w = res.shape[:2]
    h2, w2 = (130,130)
    temp = cv2.resize(res, (w2,h2), interpolation = cv2.INTER_LINEAR)
    res = cv2.resize(temp, (w, h), interpolation = cv2.INTER_NEAREST)
    img_eroded = cv2.erode(res.copy(), kernel2, iterations = 1)
    res = cv2.bitwise_and(img_eroded, img_eroded, mask = stencil)
    img_dilated1 = cv2.dilate(res.copy(), kernel, iterations = 1)
    res = cv2.resize(temp, (w,h), interpolation = cv2.INTER_NEAREST)
    img_eroded = cv2.erode(res.copy(), kernel2, iterations = 1)
    res = cv2.bitwise_and(img_eroded, img_eroded, mask = stencil)
    img_dilated1 = cv2.dilate(res.copy(), kernel, iterations = 1)
    res = cv2.bitwise_and(img_dilated1, img_dilated1, mask = stencil)
    
    h,w = res.shape[:2]
    h2, w2 = (100, 100)
    temp = cv2.resize(res, (w2, h2), interpolation = cv2.INTER_LINEAR)
    res = cv2.resize(temp, (w,h), interpolation = cv2.INTER_NEAREST)
    #res = cv2.medianBlur(res,9)
    res = (np.ceil(res/255.0)*255.0).astype('uint8')
    
    imgray = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)
   
    
    
    
    img_dilated1 = cv2.dilate(imgray, kernel2, iterations = 0)
    
   
    
    kernel3 = np.ones((12,10), np.uint8)
    img_dilated = cv2.erode(res, kernel3, iterations = 1)
   
    
    edges = cv2.Canny(img_dilated, 70, 100)
    
    return(edges, res)
