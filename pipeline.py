# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:09:18 2023

@author: guiti
"""

import os
import torch
import time
import cv2
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import fbeta_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.io import read_image
import seaborn as sn
from torch.cuda.amp import autocast, GradScaler
#Reset matplotlib parameters to avoid errors
plt.rcParams.update(plt.rcParamsDefault)

print(torch.cuda.is_available())


def plot_ROC_curve(y, probs,save_path):
        
    total_y_onehot = nn.functional.one_hot(y, num_classes=4)
    #total_probs_onehot = nn.functional.one_hot(torch.Tensor(probs).to(torch.int64), num_classes=4)
    
    for lb in range(len(np.unique(y))):
        
        fpr, tpr, _ = metrics.roc_curve(total_y_onehot[lb,:], probs[lb,:])        
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.savefig(save_path + "class_" + str(lb) + ".png")
        plt.clf()
        plt.cla()
        plt.close("all")

def plot_confusion_matrix(y, pred, save_path):

    cm = confusion_matrix(y, pred)
    plt.figure(figsize=(8,6), dpi=100)
    # Scale up the size of all text
    sn.set(font_scale = 1.1)
    
    ax = sn.heatmap(cm, annot=True,  cmap="crest")
    
    ax.set_xlabel("Predicted Type", fontsize=14, labelpad=20)
    ax.xaxis.set_ticklabels(['CNV', 'DME', 'Drusen', 'Normal'])

    ax.set_ylabel("Actual Type", fontsize=14, labelpad=20)
    ax.yaxis.set_ticklabels(['CNV', 'DME', 'Drusen', 'Normal'])

    ax.set_title("Confusion Matrix for the Atelectasis Detection Model", fontsize=14, pad=20)

    plt.savefig(save_path)
    plt.clf()
    plt.close("all")

def classwise_precision_recall_accuracy(y, pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division = 0)
    fbeta_score_ = fbeta_score(y, pred, beta = 0.5, labels = [0, 1, 2], average = None, zero_division = 0)
    return precision, recall, fscore, fbeta_score_

def gobal_precision_recall_accuracy(y, pred):
    precision, recall, fscore, _ = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], average = 'micro', zero_division = 0)
    fbeta_score_ = fbeta_score(y, pred, beta = 0.5, labels = [0, 1, 2], average = 'micro', zero_division = 0)
    return precision, recall, fscore, fbeta_score_

def check_mean_std(train_csv):
    train_mean = 0
    train_std = 0

    for line in train_csv["filename"]: 
        image = read_image("/home/vm/all_images/" + line)
        train_mean += image.float().mean().item()
        train_std += image.float().std().item()

    train_mean = train_mean / len(train_csv)
    train_std = train_std / len(train_csv)
    return train_mean, train_std