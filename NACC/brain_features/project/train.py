#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 14:37:28 2021
This script trains classifiers over NACC mri features, 
Computes feature importance and saves SHAP plots for them

It makes use of Xin's code in 
https://github.com/linbrainlab/machinelearning.git

@author: subashkhanal
"""
import os
from config import cfg
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from classifiers import classifier
from utilities import plot_SHAP, plot_ROC, save_results

featNum = 50 #Number of features taken for partial features based training
def train_NACC(features,clf,estimators,classes,repeat,data_path,results_path,plots_path):
    
    df_mri_dx = pd.read_csv(data_path)
    importance_file = pd.read_csv(os.path.join(results_path,'full_'+ classes +'_RandomForest_300.csv')).sort_values('importance',ascending= False)
    partial_cols = list(importance_file.loc[0:featNum,'Unnamed: 0'])
    df_partial = df_mri_dx.loc[:,partial_cols] # until we decide on which column to choose
    df_partial['NACCUDSD'] = df_mri_dx['NACCUDSD']
    df_full = df_mri_dx.loc[:,'NACCICV':'RTRTEMM']
    df_full['NACCUDSD'] = df_mri_dx['NACCUDSD']
    if features == 'full':
        df = df_full
        
    if features == 'partial':
        df = df_partial
    cols = df.columns[:-1]
    df_CN = df[df.NACCUDSD==1] 
    df_EMCI = df[df.NACCUDSD==2]
    df_LMCI = df[df.NACCUDSD==3]
    df_AD= df[df.NACCUDSD==4]
    
    #possible_classes = ['CN_EMCI', 'CN_LMCI', 'CN_AD', 'EMCI_LMCI','EMCI_AD', 'LMCI_AD']
    if classes == 'CN_EMCI':
        df1 = df_CN
        df2 = df_EMCI
        label1 = 1
        label2 = 2
        
    if classes == 'CN_LMCI':
        df1 = df_CN
        df2 = df_LMCI
        label1 = 1
        label2 = 3
        
    if classes == 'CN_AD':
        df1 = df_CN
        df2 = df_AD
        label1 = 1
        label2 = 4
        
    if classes == 'EMCI_LMCI':
        df1 = df_EMCI
        df2 = df_LMCI
        label1 = 2
        label2 = 3
        
    if classes == 'EMCI_AD':
        df1 = df_EMCI
        df2 = df_AD
        label1 = 2
        label2 = 4
        
    if classes == 'LMCI_AD':
        df1 = df_LMCI
        df2 = df_AD
        label1 = 3
        label2 = 4
        
        
    df_sampled = pd.concat([df1, df2])
    df_sampled = shuffle(df_sampled, random_state = 42)
    y = df_sampled.NACCUDSD
    X = df_sampled.drop('NACCUDSD',axis=1)
    
    y = label_binarize(y, classes=[label1, label2])
    
    numFeature=X.shape[1]
    y=y.ravel()
   
    scaler = StandardScaler()
    scaler.fit(X)
    X=scaler.transform(X)
    
    mean_fpr = np.linspace(0, 1, 250)
    aucs = np.zeros((repeat,5))
    acc = np.zeros((repeat,5))
    f1 = np.zeros((repeat,5))
    importance = np.zeros((repeat,5,numFeature))
    TPRS = []
    AUCS = []
    fname = features+'_'+classes+'_'+clf+'_'+str(estimators)
    for i in range(repeat):
        j = 0
        tprs = []
        aucs = []
        model = classifier(clf, estimators)
        cv = StratifiedKFold(n_splits=5,random_state = 42, shuffle = True)
        
        for train, test in cv.split(X, y):
            probas_ = model.fit(X[train], y[train]).predict_proba(X[test])
            acc[i,j] = model.score(X[test],y[test])
            y_pred = model.predict(X[test])
            f1[i,j]= f1_score(y[test], y_pred, average='weighted')
            importance[i,j,:] =  model.feature_importances_
            
            
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1],drop_intermediate='False')
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
           
            aucs.append(auc(fpr, tpr))
            
            j += 1
           
        TPRS.append(tprs)
        AUCS.append(aucs)
    
    TPRS = np.array(TPRS).mean(0) #averaged across the repeated experiments
    AUCS = np.array(AUCS).mean(0) #averaged across the repeated experiments 
    
    importance = importance.mean(0).mean(0)# Averaged across repeated experiments and folds  
    acc = acc.mean(1).mean(0) # Averaged across  folds and then over repeated experiments 
    f1 = f1.mean(1).mean(0) # Averaged across  folds and then over repeated experiments
    
    #Plot SHAP and ROC
    plot_SHAP(model,X,cols,fname,plots_path)
    plot_ROC(TPRS,AUCS,fname,plots_path)
    save_results(acc, f1, importance, cols, fname, results_path)
    
    return acc, f1, TPRS, AUCS, importance
    
                   
            
if  __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, help='Portion of features used. Options:[full, partial] ', default='full')
    parser.add_argument('--classifier', type=str, help='ML classifier to use. Options:[RandomForest, GradientBoosting] ', default='RandomForest')
    parser.add_argument('--estimators', type=int, help='number of estimators for classfiers', default=100)
    parser.add_argument('--repeat', type=int, help='number of experiments run', default=2)
    parser.add_argument('--classes', type=str, help='Binary classes to perform classification for. Options:[CN_EMCI, CN_LMCI, CN_AD, EMCI_LMCI,EMCI_AD, LMCI_AD]', default='CN_AD')
    
    args = parser.parse_args()
    
    acc, f1, tprs, aucs, importance = train_NACC(features = args.features,
               clf = args.classifier,
               estimators = args.estimators,
               classes = args.classes,
               repeat = args.repeat,
               data_path = cfg.data,
               results_path = cfg.results,
               plots_path = cfg.plots         
               )
    
    
 
