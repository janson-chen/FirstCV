
import cv2
import numpy as np
from scipy import ndimage

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
image = cv2.imread("./images/demo1.jpg")
image[:,:,2]=128
print(image.shape)
cv2.imshow("test.jpg", image)
cv2.waitKey()
cv2.destroyAllWindows()