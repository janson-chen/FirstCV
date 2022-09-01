
import cv2
import numpy as np
from scipy import ndimage
from edges.line import Line
from edges.circle import Circle
from edges.camera_capture import CameraCapure
from edges.bg_subtractor_knn import BackgroundSubtractorKNN

kernel_3x3 = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

kernel_5x5 = np.array([
    [-1, -1, -1, -1, -1],
    [-1, 1,  2,  1,  -1],
    [-1, 2,  4,  2,  -1],
    [-1, 1,  2,  1,  -1],
    [-1, -1, -1, -1, -1]
])



#image = np.zeros((1024,1024), dtype=np.uint8)
#image = cv2.pyrDown(cv2.imread("./images/demo1.jpg", cv2.IMREAD_UNCHANGED))

#ret, thresh = cv2.threshold(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
#contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#for c in contours:
#    x, y, w, h = cv2.boundingRect(c)
#    cv2.rectangle(image, (x,y), (x + w, y + h), (0, 255, 0), 2)
#    #find minimum area
#    rect = cv2.minAreaRect(c)
#    box = cv2.boxPoints(rect)
#    box = np.int0(box)
#    cv2.drawContours(image, [box], 0, (0,0,255), 3)
#cv2.imshow("contours", image)
#cv2.waitKey()
#cv2.destroyAllWindows()

# line detect
#line = Line('./images/lani.png', 50, 120)
#line.detectLines()


# circle edge detect
#circle = Circle('./images/circle.jpg')
#circle.detectCircles()


# moving objects detect
#md = CameraCapure()
#md.detectMovingObjects()

bg_sub_knn = BackgroundSubtractorKNN()
bg_sub_knn.background_subtract()


