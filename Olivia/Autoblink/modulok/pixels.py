import cv2
import numpy as np



def fullpixcount(n, cx, cy, full_pix, img):
   
    num_wpix_full = np.sum(img == 255)
    #print("fullwpix"+str(n), num_wpix_full)
    img_contour = cv2.putText(img, str(n), (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,255))
    #cv2.imshow("fullpixcount", img_contour)
    #cv2.waitKey(0)
    full_pix.append(num_wpix_full)
    return(full_pix)
def halfpixcount(n, cx, cy, stencil, half_pix, img):
    kernel = np.ones((1,1), np.uint8)
    img_dilated = cv2.erode(img.copy(), kernel, iterations = 1)
    img_dilated = cv2.bitwise_not(img.copy())
    res = cv2.bitwise_and(img_dilated, img_dilated, mask = stencil)
    h, w = res.shape[:2]
    h2, w2 = (1000, 1000)
    temp = cv2.resize(res, (w2, h2), interpolation = cv2.INTER_LINEAR)
    res = cv2.resize(temp, (w,h), interpolation = cv2.INTER_NEAREST)
    
    num_wpix_half = np.sum(res == 255)
    #print("halfwpix"+str(n), num_wpix_half)
    #half_pix.append(num_wpix_half)
    return(num_wpix_half)
