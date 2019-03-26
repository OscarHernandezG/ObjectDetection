import numpy as np
import cv2


def ApplyFilter(img, filter, threshold, image):
    rows, cols = img.shape

    img = img[:, :] / 255
    filter = filter[:, :] / 255

    kernel = np.array(filter)
    ksizex, ksizey = kernel.shape
    kradix = int(ksizex / 2)
    kradiy = int(ksizey / 2)

    # Create a copy with black padding
    pRow = int(rows + 2 * kradix)
    pCol = int(cols + 2 * kradiy)
    imgpadding = np.zeros((pRow, pCol, 1))
    imgpadding[kradix:-kradix, kradiy:-kradiy, 0] = img

    # Convolution
    matchingMap = np.array(img)
    filtered = np.zeros(img.shape)
    for i in range(0, rows - ksizex + 1):
        for j in range(0, cols - ksizey + 1):
            local = img[i:i + ksizex, j:j + ksizey]
            # if local.shape[0] == ksizex and local.shape[1] == ksizey:
            matchingMap[i, j] = np.sum((filter[:, :] - local[:, :]) ** 2)

    matchingMap = matchingMap[:, :] / matchingMap.max()

    filteredd = matchingMap.copy()
    img2 = cv2.imread(image, -1)
    color = [0, 255, 0]
    found = False;
    for i in range(0, rows - ksizex + 1):
        for j in range(0, cols - ksizey + 1):
            if filteredd[i, j] < threshold:
                found = True
                for k in range(0, 3):
                    img2[i: i + ksizex - 1, j, k] = color[k]
                    img2[i: i + ksizex - 1, j + ksizey - 1, k] = color[k]
                    img2[i, j: j + ksizey - 1, k] = color[k]
                    img2[i + ksizex - 1, j: j + ksizey - 1, k] = color[k]

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
