import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def Image_Results():
    I = [[1, 2, 3, 4], [54, 70, 142, 633, 1135]]
    for n in range(1):
        Images = np.load('Original.npy', allow_pickle=True)
        UNet = np.load('UNet.npy', allow_pickle=True)
        TransUnet = np.load('TransUnet.npy', allow_pickle=True)
        ResUnet = np.load('ResUnet.npy', allow_pickle=True)
        TransResUnet = np.load('TransResUnet.npy', allow_pickle=True)
        DYTUnetPlusPlus = np.load('3DYTUnetPlusPlus.npy', allow_pickle=True)
        for i in range(len(I[n])):
            plt.subplot(2, 3, 1)
            plt.title('Original')
            plt.imshow(Images[I[n][i]])
            plt.subplot(2, 3, 2)
            plt.title('UNet')
            plt.imshow(UNet[I[n][i]])
            plt.subplot(2, 3, 3)
            plt.title('TransUnet')
            plt.imshow(TransUnet[I[n][i]])
            plt.subplot(2, 3, 4)
            plt.title('TransResUnet')
            plt.imshow(TransResUnet[I[n][i]])
            plt.subplot(2, 3, 5)
            plt.title('ResUnet')
            plt.imshow(ResUnet[I[n][i]])
            plt.subplot(2, 3, 6)
            plt.title('3DYTUnetPlusPlus')
            plt.imshow(DYTUnetPlusPlus[I[n][i]])
            plt.tight_layout()
            plt.show()
            # cv.imwrite('./Results/Image_Results/' + 'orig-' + str(i + 1) + '.png', Images[I[n][i]])
            # cv.imwrite('./Results/Image_Results/' + 'UNet-' + str(i + 1) + '.png', UNet[I[n][i]])
            # cv.imwrite('./Results/Image_Results/' + 'TransUnet-' + str(i + 1) + '.png', TransUnet[I[n][i]])
            # cv.imwrite('./Results/Image_Results/' + 'TransResUnet-' + str(i + 1) + '.png',
            #            TransResUnet[I[n][i]])
            # cv.imwrite('./Results/Image_Results/' + 'ResUnet-' + str(i + 1) + '.png',
            #            ResUnet[I[n][i]])
            # cv.imwrite('./Results/Image_Results/' + '3DYTUnetPlusPlus-' + str(i + 1) + '.png',
            #            DYTUnetPlusPlus[I[n][i]])


if __name__ == '__main__':
    Image_Results()
