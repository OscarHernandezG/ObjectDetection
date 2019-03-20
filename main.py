import numpy as np
import cv2


def ApplyFilter(img, filter):
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
    for i in range(0, rows - ksizex):
        for j in range(0, cols - ksizey):
            local = img[i:i + ksizex, j:j + ksizey]
            if local.shape[0] == ksizex and local.shape[1] == ksizey:
                matchingMap[i, j] = np.sum((filter[:, :] - local[:, :])**2)

    matchingMap = matchingMap[:,:] / matchingMap.max()

    filteredd = matchingMap.copy()
    img2 = cv2.imread("img2.png", -1)
    color = [0, 255, 0]
    for i in range(0, rows - ksizex):
        for j in range(0, cols - ksizey):
            if filteredd[i, j] < 0.1:
                for k in range(0,3):
                    img2[i : i + ksizex, j, k] = color[k]
                    img2[i : i + ksizex, j + ksizey, k] = color[k]
                    img2[i, j : j + ksizey, k] = color[k]
                    img2[i + ksizex, j : j + ksizey, k] = color[k]

    # Show the image
    matchingMap = matchingMap[:, :] * 255
    cv2.imshow("matchingMap", np.uint8(matchingMap))
    cv2.imshow("Tu", np.uint8(img2))
    cv2.imwrite("match.png", np.uint8(matchingMap))
    return np.uint8(matchingMap)




if __name__ == '__main__':

    img2 = cv2.imread("img2.png", 0)
    t1img2 = cv2.imread("t1-img2.png", 0)

    ApplyFilter(img2, t1img2)
    cv2.waitKey()



