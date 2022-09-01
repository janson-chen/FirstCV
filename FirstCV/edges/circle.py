import cv2
import numpy as np


class Circle(object):
    def __init__(self, src) -> None:
        self.src = src


    def detectCircles(self):
        img = cv2.imread(self.src)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_img = cv2.medianBlur(gray, 7)
        #cimg = cv2.cvtColor(blur_img, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(blur_img, cv2.HOUGH_GRADIENT, 1, 200, param1 = 100, param2 = 30, minRadius = 0, maxRadius = 0)
        circles = np.uint16(np.around(circles))

        for i in circles[0,:]:
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

        cv2.imwrite("planets_circles.jpg", img)
        cv2.imshow("HoughCircles", img)
        cv2.waitKey()
        cv2.destroyAllWindows()




