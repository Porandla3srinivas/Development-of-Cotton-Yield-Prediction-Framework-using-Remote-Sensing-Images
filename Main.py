import os
import random as rn
import numpy as np
import cv2 as cv
from numpy import matlib
from BWO import BWO
from Batch_Split import Batch_Split
from Global_Vars import Global_Vars
from HCO import HCO
from Image_Results import *
from MRA import MRA
from Model_3DYTUnetPlusPlus import Model_3DYTUnetPlusPlus
from Model_CNN import Model_CNN
from Model_FCM import Model_FCM
from Model_RAN import Model_RAN
from Model_VGG16 import Model_VGG16
from Model_ViT_SNetV2 import Model_ViT_SNetV2
from NRO import NRO
from Objective_Function import objfun_cls
from Plot_Results import *
from Proposed import Proposed

# Read Dataset
an = 0
if an == 1:
    images = []
    dir = './Dataset/'
    dir1 = os.listdir(dir)
    for i in range(len(dir1)):
        file = dir + dir1[i]
        read = cv.imread(file)
        read = cv.resize(read, [700, 700])
        images.append(read)
    np.save('Original.npy',images)

# Read Ground_Truth
an = 0
if an == 1:
    dir = './Dataset/'
    dir1 = os.listdir(dir)
    imgs=[]
    imgs1 = []
    for i in range(len(dir1)):
        file = dir+dir1[i]
        read = cv.imread(file)
        read = cv.cvtColor(read,cv.COLOR_RGB2GRAY)
        read = cv.resize(read,[700,700])
        imgs.append(read)
    imgs = np.asarray(imgs)
    feat = Model_FCM(imgs)
    np.save('Ground_Truth.npy',feat)

# Read Target
an = 0
if an == 1:
    targets = []
    for i in range(4):
        Images = np.load('Ground_Truth.npy', allow_pickle=True)[i]
        patche = Batch_Split(Images)
        Target = []
        for j in range(len(patche)):
            gr_tru = patche[j]
            uniq = np.unique(gr_tru)
            lenUniq = [len(np.where(gr_tru == uniq[k])[0]) for k in range(len(uniq))]
            maxIndex = np.where(lenUniq == np.max(lenUniq))[0][0]
            target = uniq[maxIndex]
            Target.append(target)
        Targ = np.asarray(Target)
        uni = np.unique(Targ)
        tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
        for a in range(len(uni)):
            ind = np.where((Targ == uni[a]))
            tar[ind[0], a] = a
        targets.append(tar)
    np.save('Targets.npy', targets)

# Segmentation
an = 0
if an == 1:
    Images = np.load('Original.npy',allow_pickle=True)
    Gt = np.load('Ground_Truth.npy',allow_pickle=True)
    Seg = Model_3DYTUnetPlusPlus(Images,Gt)
    np.save('Segmented.npy',Seg)

# Optimization - for Classification
an = 0
if an == 1:
    Images = np.load('Segmented.npy', allow_pickle=True)
    Tar = np.load('Targets.npy', allow_pickle=True)
    Npop = 10
    Chlen = 3
    Global_Vars.Images = Images
    Global_Vars.Target = Tar
    xmin = matlib.repmat([5, 5, 50], Npop, 1)
    xmax = matlib.repmat([255, 50, 250], Npop, 1)
    initsol = np.zeros((xmax.shape))
    for p1 in range(Npop):
        for p2 in range(xmax.shape[1]):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun_cls
    Max_iter = 25

    print("EOO...")
    [bestfit1, fitness1, bestsol1, time1] = BWO(initsol, fname, xmin, xmax, Max_iter)  # BWO

    print("MRA...")
    [bestfit2, fitness2, bestsol2, time2] = MRA(initsol, fname, xmin, xmax, Max_iter)  # MRA

    print("NRO...")
    [bestfit3, fitness3, bestsol3, time3] = NRO(initsol, fname, xmin, xmax, Max_iter)  # NRO

    print("HCO...")
    [bestfit4, fitness4, bestsol4, time4] = HCO(initsol, fname, xmin, xmax, Max_iter)  # HCO

    print("Proposed..")
    [bestfit5, fitness5, bestsol5, time5] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    sols = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('bestsol.npy', sols)

# Classification
an = 0
if an == 1:
    Feat = np.load('Segmented.npy', allow_pickle=True)
    Tar = np.load('Targets.npy', allow_pickle=True)
    bests = np.load('bestsol.npy', allow_pickle=True)
    Eval_all = []
    Optimizer = ['Adam', 'SGD', 'RMSProp', 'Adadelta', 'AdaGrad']
    for m in range(len(Optimizer)):  # for all learning percentage
        EVAL = np.zeros((10, 14))
        per = round(len(Feat) * 0.75)
        Train_Data = Feat[:per, :, :]
        Train_Target = Tar[:per, :]
        Test_Data = Feat[per:, :, :]
        Test_Target = Tar[per:, :]
        for j in range(bests.shape[0]):  # for all algorithms
            soln = bests[j]
            EVAL[j, :], pred1 = Model_ViT_SNetV2(Feat,Tar, Optimizer[m], soln)  # with Optimization Vit with sufflenetv2DTCN
        EVAL[5, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer[m])  # CNN Model
        EVAL[6, :], pred3 = Model_RAN(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer[m])  # RAN model
        EVAL[7, :], pred4 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target, Optimizer[m])  # Vgg16 model
        EVAL[8, :], pred5 = Model_ViT_SNetV2(Feat,Tar,Optimizer[m])  # without Optimization Vit with sufflenetv2
        EVAL[9, :] = EVAL[4, :]  # with Optimization DTCN
        Eval_all.append(EVAL)
    np.save('Eval_all.npy', Eval_all)


plot_results_optimizer()
plot_results()
plot_Segmentation_results_1()
plotConvResults()
Plot_ROC_Curve()
Image_Results()