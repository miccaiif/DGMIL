from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
import time
import logging
import argparse
from collections import OrderedDict
import faiss

import torch
import torch.nn as nn
from tqdm import tqdm
from models import SupResNet, SSLResNet
from utils import (
    get_features,
    get_roc_sklearn,
    get_pr_sklearn,
    get_fpr,
    get_scores_one_cluster,
)
import sklearn.metrics as skm
import data

# local utils for evaluation
def get_scores(ftrain, ftest, food, clusters):
    if clusters == 1:
        return get_scores_one_cluster(ftrain, ftest, food)
    else:
        ypred = get_clusters(ftrain, clusters)
        return get_scores_multi_cluster(ftrain, ftest, food, ypred)


def get_clusters(ftrain, nclusters):
    kmeans = faiss.Kmeans(
        ftrain.shape[1], nclusters, niter=100, verbose=False, gpu=False
    )
    ftrain = ftrain.astype(np.float32)
    kmeans.train(np.random.permutation(ftrain))
    _, ypred = kmeans.assign(ftrain)
    return ypred


def get_scores_multi_cluster(ftrain, ftest, food, ypred):
    xc = [ftrain[ypred == i] for i in np.unique(ypred)]

    din = [
        np.sum(
            (ftest - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (ftest - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]
    dood = [
        np.sum(
            (food - np.mean(x, axis=0, keepdims=True))
            * (
                np.linalg.pinv(np.cov(x.T, bias=True)).dot(
                    (food - np.mean(x, axis=0, keepdims=True)).T
                )
            ).T,
            axis=-1,
        )
        for x in xc
    ]

    din = np.min(din, axis=0)
    dood = np.min(dood, axis=0)

    return din, dood

def get_eval_results_score(ftrain, ftest, food, clusters):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food, clusters)
    #Calculate the distance of each data point in the test and the distance of each data point in the ood, respectively.
    fpr95 = get_fpr(dtest, dood)
    auroc, aupr = get_roc_sklearn(dtest, dood), get_pr_sklearn(dtest, dood)
    return fpr95, auroc, aupr
    # return dtest,dood

def get_eval_results_dis_feats(ftrain, ftest, food, clusters):
    """
    None.
    """
    # standardize data
    ftrain /= np.linalg.norm(ftrain, axis=-1, keepdims=True) + 1e-10
    ftest /= np.linalg.norm(ftest, axis=-1, keepdims=True) + 1e-10
    food /= np.linalg.norm(food, axis=-1, keepdims=True) + 1e-10

    m, s = np.mean(ftrain, axis=0, keepdims=True), np.std(ftrain, axis=0, keepdims=True)

    ftrain = (ftrain - m) / (s + 1e-10)
    ftest = (ftest - m) / (s + 1e-10)
    food = (food - m) / (s + 1e-10)

    dtest, dood = get_scores(ftrain, ftest, food, clusters)
    #Calculate the distance of each data point in the test and the distance of each data point in the ood, respectively.

    return dtest,dood

def get_score_and_dis(num_cluster,seed,neg_feats_training,neg_feats_testing,pos_neg_feats_testing,pos_pos_feats_testing):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    all_test_neg = np.vstack((neg_feats_testing,pos_neg_feats_testing))

    fpr95, auroc, aupr = get_eval_results_score(
        np.copy(neg_feats_training),
        np.copy(all_test_neg),
        np.copy(pos_pos_feats_testing), num_cluster)

    return auroc

def get_score_and_dis_MAE(num_cluster,seed,neg_feats_training,neg_feats_testing,pos_feats_testing):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    fpr95, auroc, aupr = get_eval_results_score(
        np.copy(neg_feats_training),
        np.copy(neg_feats_testing),
        np.copy(pos_feats_testing), num_cluster)

    return auroc

def get_score_and_dis_feats(num_cluster,seed,neg_feats_training,pos_feats_training):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    dis_neg_train,dis_pos_train = get_eval_results_dis_feats(
        np.copy(neg_feats_training),
        np.copy(neg_feats_training), #Calculate the more abnormal inside itself
        np.copy(pos_feats_training), num_cluster)

    return dis_neg_train, dis_pos_train

def get_auc_score(labels,data):
    auroc = skm.roc_auc_score(labels, data)
    return auroc


