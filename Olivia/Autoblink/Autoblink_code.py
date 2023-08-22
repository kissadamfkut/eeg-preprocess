import cv2
import numpy as np
import mne


# modulok importalasa
from Olivia.Autoblink.modulok import mneproba as mnes
from Olivia.Autoblink.modulok import maszkolas as maszk
from Olivia.Autoblink.modulok import konturozas as cntss
from Olivia.Autoblink.modulok import pixels as pix
from Olivia.Autoblink.modulok import kepdil as dil


def remove_blink(inraw):

    # fajl beolvasasa, ica lefuttatasa, kep kimentese
    img, raw, ica = mnes.generalfromraw(inraw)

    #cv2.imshow('image', img)
    #cv2.waitKey(0);


    #szurkearnyalatos kep generalas
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray', gray)
    #cv2.waitKey(0);
    #kep elmosasa
    gray_blur = cv2.medianBlur(gray, 3)
    #cv2.imshow('gray_blur', gray_blur)
    #cv2.waitKey(0);

    #korok detektalas
    rows = img.shape[0]
    edges1 = cv2.Canny(gray_blur, 70, 100)
    #cv2.imshow("edge1", edges1)
    #cv2.waitKey(0)
    circles = cv2.HoughCircles(edges1, cv2.HOUGH_GRADIENT, 1, rows / 16,
                            param1=90, param2=15,
                            minRadius=52, maxRadius=58)

    #szin maszk keszitese (sotetkek-piros)
    masked_img = maszk.mask(img, img, lower_val = np.array([105,0,0]), method = 'add')
    #cv2.imshow("masked_img", masked_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # korvonal rajzolasa pirossal
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            center = (i[0], i[1])
            print(center, i[2])
            # circle center
            #cv2.circle(final, center, 1, (0, 0, 0), 1)
            # circle outline
            radius = i[2]
            circled_img = cv2.circle(masked_img, center, radius, (0, 0, 255), 2)


    #cv2.imshow("detected circle", circled_img)
    #cv2.waitKey(0)

    # korvonalakra keszitett maszk -> konturozas miatt fontos
    circle_mask = maszk.mask(circled_img, circled_img, lower_val = np.array([0,0,255]), method = 'add')

    #cv2.imshow("circle_mask", circle_mask)
    #cv2.waitKey(0)

    # korok konturozasa, hierarchy rendezes, konturok sorbarendezese
    imgray = cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoured_img = cv2.drawContours(circle_mask.copy(), contours, -1, (0,255,0), 2)

    #cv2.imshow("contoured_img", contoured_img)
    #cv2.waitKey(0)

    contour = cntss.contour(contours, hierarchy)

    # fekete-feher korok konturozva
    cnt_bw_img = maszk.mask(contoured_img, circled_img, lower_val = np.array([0,0,255]), method = 'subtract')
    #cv2.imshow("cnt_bw_img", cnt_bw_img)
    #cv2.waitKey(0)

    # valtozok deklaralasa
    full_pix = []
    half_pix = []
    bad_circles = []
    ica_number = []

    half_bw_circle = cnt_bw_img.copy()


    #felkorok kivitelezese ujabb nagyobb de lentebbi kozeppontu korok egymasra illesztesevel
    for n, cnt in enumerate(contour):
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(half_bw_circle, (cx, (cy+25)), 65, (255,255,255), -1)
    #cv2.imshow("halfbwcircle", half_bw_circle)
    #cv2.waitKey(0)

    #a teljes kor pixelszamat vizsgalom
    for n, cnt in enumerate(contour):
        stencil = np.zeros(img.shape[:2], dtype = 'uint8')
        stencil = cv2.drawContours(stencil.copy(), contour, n, (255,255,255), -1)
        single_img = cv2.drawContours(cnt_bw_img.copy(), contour, n, (255,255,255), -1)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        full_pix = pix.fullpixcount(n,cx, cy, full_pix, single_img)

    # masodik legnagyobb pixelszam ertekenek megvalasztasa
    largest_full_pix = max(full_pix)
    full_pix.remove(largest_full_pix)
    seclarg_full_pix = max(full_pix)


    # pixelszam alapjan torteno szelekcio
    for n, cnt in enumerate(contour):
        stencil = np.zeros(img.shape[:2], dtype = 'uint8')
        stencil = cv2.drawContours(stencil.copy(), contour, n, (255,255,255), -1)
        single_img = cv2.drawContours(cnt_bw_img.copy(), contour, n, (255,255,255), -1)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        num_wpix_half = pix.halfpixcount(n, cx, cy, stencil, half_pix, half_bw_circle)
        num_wpix_full = np.sum(single_img == 255)
        
        #cv2.imshow("fullpixcount", single_img)
        #cv2.waitKey(0)
        
        if num_wpix_full >= seclarg_full_pix or num_wpix_half<600:
            bad_circles.append((cx,cy))

    # a rossz korok kitakarasa egy ujabb nagyobb kor raillesztesevel
    for n in bad_circles:
        cv2.circle(cnt_bw_img, (n), 65, (255,255,255), -1)
    #cv2.imshow("vegleges", cnt_bw_img)
    #cv2.waitKey(0)


    #korok dilatacioja, erozioja, egyenesillesztes elvegzese
    #ica kivalasztasa egyenesillesztes alapjan
    for n, cnt in enumerate(contour):
        stencil = np.zeros(img.shape[:2], dtype = 'uint8')
        stencil = cv2.drawContours(stencil.copy(), contour, n, (255,255,255), -1)
        single_img = cv2.drawContours(cnt_bw_img.copy(), contour, n, (255,255,255), -1)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        edges, res = dil.picdil(n, stencil, cnt_bw_img)
        #edges, res = dilk.picdil(n, stencil, cnt_bw_img)
        
        
        
        #cv2.imshow("edge"+str(n),edges)
        #cv2.waitKey(0) 


        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 45, maxLineGap = 10)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                #cv2.line(res, (x1,y1), (x2,y2), (0,255,0), 4)
                szam = n
                if n not in ica_number:
                    ica_number.append(szam)
            
        #cv2.imshow("res_lesjs"+str(n),res)
        #cv2.waitKey(0)
            
    ica.exclude = ica_number
    return ica
