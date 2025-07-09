import numpy as np


def Batch_Split(image, batchSize=None):
    if batchSize is None:
        batchSize = [8, 8]
    row = image.shape[0]
    col = image.shape[1]
    row = batchSize[0] - (row % batchSize[0]) + row
    col = batchSize[1] - (col % batchSize[1]) + col
    image = np.resize(image, (row, col))
    noOfRow = round(row / batchSize[0])
    noOfCol = round(col / batchSize[1])
    splittedImage = []
    for i in range(noOfRow):
        for j in range(noOfCol):
            splittedImage.append(image[i * batchSize[0]:(i + 1) * batchSize[0], j * batchSize[1]:(j + 1) * batchSize[1]])
    return splittedImage


if __name__ == '__main__':
    image = np.arange(500 * 500)
    image = np.reshape(image, (500, 500))
    Batch_Split(image)



