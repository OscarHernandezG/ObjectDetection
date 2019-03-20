import numpy as np
import cv2


if __name__ == '__main__':

    img1 = cv2.imread("img1.png", 0)

    cv2.imshow("img1", img1)

    cv2.waitKey()