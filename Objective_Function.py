import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_ViT_SNetV2 import Model_ViT_SNetV2


def objfun_cls(Soln):
    Feat = Global_Vars.Feat
    Tar = Global_Vars.Target
    Tar = np.reshape(Tar, (-1, 1))
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Feat.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = Model_ViT_SNetV2(Feat, Tar, sol)
            Fitn[i] = (1 / Eval[4]) + Eval[8]
        return Fitn
    else:
        learnper = round(Feat.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = Model_ViT_SNetV2(Feat, Tar, sol)
        Fitn = (1 / Eval[4]) + Eval[8]
        return Fitn
