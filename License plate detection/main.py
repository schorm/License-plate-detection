import cv2
import imutils
import numpy as np
import os
def detect(id,check):
    #read images
    img = cv2.imread(f'D:/Project/openCV/License plate detection/input/{id}.jpg') 

    # Resize images
    img = cv2.resize(img, (620, 480))
    #Grayscale map
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Bilateral filtering
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    #Gaussian Blur
    img_blur=cv2.GaussianBlur(gray,(7,5),0,0,cv2.BORDER_DEFAULT)
    #cv2.imshow('img1', img_blur)

    #Morphological processing
    kernel=np.ones((13,13),np.uint8)
    img_opening = cv2.morphologyEx(img_blur, cv2.MORPH_OPEN, kernel)
    # cv2.imshow('img_opening1',img_opening)
    img_opening = cv2.addWeighted(img_blur, 1, img_opening, -1, 0)
    #cv2.imshow('img_opening2',img_opening)

    #Threshold segmentation

    ret,img_thre=cv2.threshold(img_opening,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #Edge Detect
    edged = cv2.Canny(img_thre, 30, 200)
    #cv2.imshow('img2', edged)
    kernel=np.ones((23,23),np.uint8)
    img_edge1 = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow('img2', img_edge1)
    kernel=np.ones((23,23),np.uint8)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    #cv2.imshow('img1', img_edge2)

    #Find Rectangle
    contours = cv2.findContours(img_edge2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #Used in conjunction with the above sentence: for compatibility with cv2 and cv3
    contours = imutils.grab_contours(contours)
    temp_contours = []
    for contour in contours:
        #Extracts area of specified size
        if 20000>cv2.contourArea( contour ) > 5000 :
                temp_contours.append(contour)
    car_plate = []
    for temp_contour in temp_contours:
            rect_tupple = cv2.minAreaRect( temp_contour )
            rect_width, rect_height = rect_tupple[1]
            if rect_width < rect_height:
               rect_width, rect_height = rect_height, rect_width
            aspect_ratio = rect_width / rect_height

            # License plates normally have a width to height ratio between 2 - 5.5
            if aspect_ratio > 2 and aspect_ratio < 5 and rect_tupple[2]<90:
                car_plate.append( temp_contour )
                rect_vertices = cv2.boxPoints( rect_tupple )
                rect_vertices = np.int0( rect_vertices )
            if len(car_plate)==0:
                if aspect_ratio > 1 and aspect_ratio < 5 :
                    car_plate.append( temp_contour )
                    rect_vertices = cv2.boxPoints( rect_tupple )
                    rect_vertices = np.int0( rect_vertices )
    #extract car_plates
    for car_plates in car_plate:
        row_min,col_min = np.min(car_plates[:,0,:],axis=0)
        row_max, col_max = np.max(car_plates[:, 0, :], axis=0)
        cv2.rectangle(img, (row_min,col_min), (row_max, col_max), (0,255,0), 2)
        card_img = img[col_min:col_max,row_min:row_max,:]
        cv2.imwrite( f'D:/Project/openCV/License plate detection/output/car_plate{id}.jpg', card_img)

    # show images
    if check:
        cv2.imshow('img', img)
        cv2.imshow('gray', gray)
        cv2.imshow('edged', edged)
        cv2.imshow('img_edge2',img_edge2)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()





def main():
    a=list(range(1,11))
    for i in a:
     detect(i,False)
    print("!!")



if __name__ == '__main__':
    # code in this block will only be executed when the script is run as the main program
    main()