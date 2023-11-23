# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 10:15:23 2023

@author: guiti
"""

from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
import pandas as pd

def create_cross_validation_set(old_csv_dir, csv_dir, set_number, delim):               
    ov_annotations = pd.read_csv(csv_dir , delimiter=delim)
    old_csv = pd.read_csv(old_csv_dir, delimiter=delim)
    group_kfold = StratifiedGroupKFold(n_splits=5)
    group_kfold.get_n_splits(old_csv["filename"], old_csv["label"])
    print()
    print()
    train_annotations = pd.Series([], dtype= "int8")
    val_annotations = pd.Series([], dtype= "int8")

    print("Separate Trainning and Testing -------------------------------------")
    for i, (train_index, test_index) in enumerate(group_kfold.split(old_csv["filename"], old_csv['label'], old_csv['patient'])):
        print(f"Fold {i}:") 
        print("Train Class -------------------------------------------------------") 
        
        fold_train_annotations = old_csv.iloc[train_index]
        
        print("Val Class ---------------------------------------------------------")
        fold_val_annotations = old_csv.iloc[test_index]

        fold_train_annotations.insert(0, 'Fold', i) 
        fold_val_annotations.insert(0, 'Fold', i) 

        train_annotations = pd.concat([train_annotations, fold_train_annotations])
        val_annotations = pd.concat([val_annotations, fold_val_annotations])

    fold_val_annotation = train_annotations[train_annotations['Fold'] == 0]
    
    dme_annotations = fold_val_annotation[fold_val_annotation["label"]==2][0:881]
    normal_annotations = fold_val_annotation[fold_val_annotation["label"]==4][0:881]
    dme_annotations["label"] = 1
    normal_annotations["label"] = 3

    new_csv = pd.concat([dme_annotations, normal_annotations])
    new_csv = pd.concat([new_csv, ov_annotations])
    
    group_kfold = StratifiedGroupKFold(n_splits=set_number)
    group_kfold.get_n_splits(new_csv["filename"], new_csv["label"])

    print("Separate Trainning and Validation -------------------------------------")
    for i, (train_index, test_index) in enumerate(group_kfold.split(new_csv["filename"], new_csv['label'], new_csv['patient'])):
        print(f"Fold {i}:") 
        print("Train Class -------------------------------------------------------") 
        
        train_annotations = new_csv.iloc[train_index]

        print("Val Class ---------------------------------------------------------")
        val_annotations = new_csv.iloc[test_index]
        
        #fold_train_annotations['Fold'] = i
        #fold_val_annotations['Fold'] = i 
        
        #train_annotations = pd.concat([train_annotations, fold_train_annotations])
        #val_annotations = pd.concat([val_annotations, fold_val_annotations])
        print("new train label values:" ,  train_annotations.label.value_counts())
        print("new val label values:" ,  val_annotations.label.value_counts())


    return  train_annotations, val_annotations
