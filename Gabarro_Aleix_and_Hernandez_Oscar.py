import numpy as np
import cv2


def ApplyFilter(img, filter, threshold, image):

    img = img[:, :] / 255
    filter = filter[:, :] / 255

    ksizex, ksizey = filter.shape

    x = img.shape[0] - ksizex + 1
    y = img.shape[1] - ksizey + 1
    matchingMap = np.zeros((x, y))
    for i in range(0, x):
        for j in range(0, y):
            local = img[i:i + ksizex, j:j + ksizey]
            matchingMap[i, j] = np.sum((filter[:, :] - local[:, :]) ** 2)

    matchingMap = matchingMap[:, :] / matchingMap.max()

    img2 = cv2.imread(image, -1)
    found = False
    for i in range(0, x):
        for j in range(0, y):
            if matchingMap[i, j] < threshold:
                found = True
                cv2.rectangle(img2, (j - 1, i - 1), (j + ksizex, i + ksizey), [0, 255, 0], 1)

    # Show the image
    matchingMap = matchingMap[:, :] * 255
    cv2.imshow("matchingMap", np.uint8(matchingMap))
    cv2.imshow("Input image", np.uint8(img2))
    cv2.imwrite("match.png", np.uint8(matchingMap))
    return found


if __name__ == '__main__':

    image = input("Input image: ")
    target = input("Target image: ")
    threshold = float(input("Detection Threshold: "))

    img = cv2.imread(image, 0)
    trgt = cv2.imread(target, 0)
    if img is not None and trgt is not None:

        imgFound = np.zeros((40, 260, 3), np.uint8)

        font = cv2.FONT_HERSHEY_TRIPLEX
        if ApplyFilter(img, trgt, threshold, image):
            cv2.putText(imgFound, "TARGET FOUND", (5, 30), font, 1, (0, 255, 0), 2)
        else:
            imgFound = np.zeros((40, 340, 3), np.uint8)
            cv2.putText(imgFound, "TARGET NOT FOUND", (5, 30), font, 1, (255, 255, 0), 2)

        cv2.imshow("Result", imgFound)

        colorImg = cv2.imread(image, -1)
        colorTrgt = cv2.imread(target, -1)

        cv2.imshow("Target", colorTrgt)

    else:
        if img is None:
            print("Input image '" + image + "' not found")
        if trgt is None:
            print("Target image '" + target + "' not found")

    cv2.waitKey()
