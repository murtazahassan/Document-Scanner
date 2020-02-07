import time
import cv2
import numpy as np
from Suduko import myUtlis


########################################################################
webCamFeed = True
pathImage = "4.jpg"
cap = cv2.VideoCapture(0)
heightImg = 1280
widthImg  = 710

########################################################################

myUtlis.initializeTrackbars()
count=0

while True:

    ##############   FINDING THE

    # PREPARE THE IMAGE
    success, img = cap.read()
    img = cv2.resize(img, (widthImg, heightImg)) # RESIZE IMAGE TO MAKE IT A SQUARE IMAGE
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # CREATE A BLANK IMAGE FOR TESTING DEBUGING IF REQUIRED
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # CONVERT IMAGE TO GRAY SCALE
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1) # ADD GAUSSIAN BLUR
    #imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, 1, 1, 11, 2) # APPLY ADATIVE THRESHOLD
    thres=myUtlis.valTrackbars()
    imgThreshold = cv2.Canny(imgBlur,thres[0],thres[1])
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    ## FIND ALL COUNTOURS
    imgContours = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    imgBigContour = img.copy() # COPY IMAGE FOR DISPLAY PURPOSES
    contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # FIND ALL CONTOURS
    cv2.drawContours(imgContours, contours, -1, (0, 255, 0), 3) # DRAW ALL DETECTED CONTOURS


     # FIND THE BIGGEST COUNTOUR AND USE IT AS SUDUKO
    biggest, maxArea = myUtlis.biggestContour(contours) # FIND THE BIGGEST CONTOUR
    if biggest.size != 0:
        biggest=myUtlis.reorder(biggest)
        cv2.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 10) # DRAW THE BIGGEST CONTOUR
        pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
        matrix = cv2.getPerspectiveTransform(pts1, pts2) # GER
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
        imgDetectedDigits = imgBlank.copy()
        imgWarpGray = cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 11, 2) # APPLY ADATIVE THRESHOLD
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv2.medianBlur(imgAdaptiveThre,3)
        imageArray = ([img,imgDial,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])
        stackedImage = myUtlis.stackImages(imageArray, 0.25)
        cv2.imshow('Stacked Images', stackedImage)
    #
    #
    else:

        imageArray = ([img, imgThreshold, imgContours, imgBigContour],
                      [imgBlank, imgBlank, imgBlank, imgBlank])
        stackedImage = myUtlis.stackImages(imageArray, 0.25)

        cv2.imshow('Stacked Images', stackedImage)



    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("Scanned/myImage"+str(count)+".jpg",imgAdaptiveThre)
        count += 1
        cv2.putText(img, "Scan Saved", (100, 100), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
        imageArray = ([img, imgDial, imgThreshold, imgContours],
                      [imgBigContour, imgWarpColored, imgWarpGray, imgAdaptiveThre])
        stackedImage = myUtlis.stackImages(imageArray, 0.25)
        cv2.imshow('Stacked Images', stackedImage)
        cv2.waitKey(1000)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
