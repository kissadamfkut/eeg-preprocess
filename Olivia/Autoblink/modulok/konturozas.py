import cv2
import numpy as np

def contour(contours, hierarchy):
    contour = list()
    hierarchy = hierarchy[0]
    for component in zip(contours, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        #x,y,w,h = cv2.boundingRect(currentContour)
        if currentHierarchy[2] < 0:
            contour.append(currentContour)
    
    #konturok sorbarendezese
    contour = tuple(contour)
    def sort_contours(cnts, method="left-to-right"):
    	# initialize the reverse flag and sort index
    	reverse = False
    	i = 0
    	# handle if we need to sort in reverse
    	if method == "right-to-left" or method == "bottom-to-top":
    		reverse = True
    	# handle if we are sorting against the y-coordinate rather than
    	# the x-coordinate of the bounding box
    	if method == "top-to-bottom" or method == "bottom-to-top":
    		i = 1
    	# construct the list of bounding boxes and sort them from top to
    	# bottom
    	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
    	# return the list of sorted contours and bounding boxes
    	return (cnts)
    
    cntsor1 = sort_contours(contour[0:5], "left-to-right")
    cntsor2 = sort_contours(contour[5:10], "left-to-right")
    cntsor3 = sort_contours(contour[10:15], "left-to-right")
    contour = sort_contours(cntsor3) + sort_contours(cntsor2) + sort_contours(cntsor1)
    return contour