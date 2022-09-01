import cv2
import numpy as np


class Line(object):
    def __init__(self, src, thread1, thread2) -> None:
        self.src = src
        self.minLineLength = 20
        self.maxLineGap = 5
        self.thread1 = thread1
        self.thread2 = thread2


    def detectLines(self):
        img = cv2.imread(self.src)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, self.thread1, self.thread2)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, self.minLineLength, self.maxLineGap)
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("edges", edges)
        cv2.imshow("lines", img)
        cv2.waitKey()
        cv2.destroyAllWindows()

