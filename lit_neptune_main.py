# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:20:41 2023

@author: gbarbosa
"""
import lightning.pytorch as pl
import utils
import torch
from torch.utils.data import DataLoader
import numpy as np
import gc
import pandas as pd
import os
import torch.nn as nn
import torchvision
from torch.multiprocessing import Pool, Process, set_start_method
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from neptune.version import version as neptune_client_version
from lightning import Fabric
import lightning as L
from lightning.pytorch.tuner import Tuner
import torchvision.models as models
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
#try:
#     set_start_method('spawn')
#except RuntimeError:
#    pass

from torch.utils.data import WeightedRandomSampler

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

torch.set_float32_matmul_precision("medium")
torch.cuda.empty_cache()
print("Brain Float 16 cuda support: ")
torch.cuda.is_bf16_supported()
fabric = Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
fabric.launch()
if __name__ == "__main__":
    print("Main File")
    
    # your credentials
    # Define settings for reproducibility 
    torch.backends.cudnn.benchmark = False
    def seed_worker(worker_seed):
        torch.cuda.manual_seed(worker_seed)

    g = torch.Generator(device='cpu')
    g.manual_seed(4)
    seed_worker(4)

    def count_model_weights(model):
        weights = 0
        for parameter in model.parameters():
            weights += torch.flatten(parameter.data).size()[0]
        return weights
    
    params = {"init_learning_rate": 0.00003, "optimizer": "Adam", "image_resize" : 224, "patience" : 10, "plateu" : "static_lr", "threshold_reduce_on_plateu" : 1e-3, "dropout" : 0, "database" : "Kermany", "probability function" : "log_softmax", "classes" : 3, "loss" : "cross_entropy", "pre_trained" : "yes_with public oct db", "frozen_features " : True} #, "rand_classifier": True}
    
    num_classes = 3
    num_GPU = 1
    learning_rate = 0.00001
    pat = 10 # number of epochs after validation maximum until the model stops trainning
    beta = 0.5 # beta value for the f-beta function
    #plat = 10 # number of epochs to change the learning rate 
    
    fold_num = 10 #fold num tem de ser maior que 1
    ov_train_csv_dir = "/home/vm/ov_df_rest.csv"
    train_df =  "/home/vm/train_dataframe.csv"

    img_dir = "/home/vm/all_images/"
    model_path = "/home/vm/model/"
    metrics_path = "/home/vm/metrics/"

    BATCH_SIZE_dict = [16, 32, 64]
    
    train_annotations, val_annotations = utils.create_cross_validation_set(train_df, ov_train_csv_dir, fold_num, ",")
    test_annotations = pd.read_csv("/home/vm/new_test_df2.csv")
    resize_crop = torchvision.transforms.Resize((224, 224))        

    for BATCH_SIZE in BATCH_SIZE_dict:
            print("Batch Size -------------- ", BATCH_SIZE) 
        
            
            print("Model in Training: ", "vgg19")
            #for fold in range(np.unique(val_annotations["Fold"].to_numpy()).size):

            model = models.vgg19()#"weights= 'IMAGENET1K_V1')
            # classifier_rand = nn.Sequential(
            #         nn.Linear(25088, 4096),
            #         nn.ReLU(inplace=True),
            #         nn.Dropout(p=0.5),
            #         nn.Linear(4096, 4096),
            #         nn.ReLU(inplace=True),
            #         nn.Dropout(p=0.5),
            #         nn.Linear(4096, 4))
            model.classifier[6] = nn.Linear(4096, 4)
            new_key_list = []
            checkpoint = torch.load("/home/vm/model/epoch=47-step=59520.ckpt")
            for key in checkpoint["state_dict"].keys(): 
                    new_key_list.append(key.replace("model.", ""))
            final_dict = dict(zip(new_key_list, list(checkpoint["state_dict"].values())))
            # model.classifier = classifier_rand
            model.load_state_dict(final_dict)
            model.classifier[6] = nn.Linear(4096, 3)

            for param in model.features.parameters():
                param.requires_grad = False

            neptune_logger = NeptuneLogger(
                                     project="gbarbosa/OCT-venous-occlusion",
                                     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJiNWY5YmU5Mi1kNzJlLTRmYzItOTBjNi01NDU4YzlkNTdkYTAifQ=="
                    )

            params["model"] = "vgg19"
            params["batch_size"] = BATCH_SIZE
            params["fold"] = 1
            params["preprocessing"] = "resize_only_from_crop_top"
                    
            print("Trainning Fold Number: ", 1)
            print() 
                    
            total_train_metrics = pd.DataFrame(columns= ["Train Loss", "Train AUC", "Train Precision" , "Train Recall", "F1 Score", "Fb Score", "Train Duration"]) 
            total_val_metrics = pd.DataFrame(columns= ["Val Loss", "Val AUC", "Val Precision" , "Val Recall", "F1 Score", "Fb Score", "Val Duration"]) 
            best_val_fbeta = 0
            best_val_loss = 1000
            flag = 0
            epoch = 0 
            #train_fold_annotations = train_annotations.loc[train_annotations['Fold'] == 1]
            #val_fold_annotations = val_annotations.loc[val_annotations['Fold'] == 1]
            
            print("train label values fold 1:" ,  train_annotations.label.value_counts())
            print("val label values fold 1:" ,  val_annotations.label.value_counts())

            print("---------------------------Train---------------------------")        
            n1 = sum(train_annotations["label"] == 1.0)
            print("Label DME", n1)
            params["Train_Label DME"] = n1
            n2 = sum(train_annotations["label"] == 2.0)
            print("Label OV", n2)
            params["Train_Label OV"] = n2
            n3 = sum(train_annotations["label"] == 3.0)
            print("Label Normal", n3)
            params["Train_Label Normal"] = n3

            print("---------------------------Validation---------------------------")        
            n1 = sum(val_annotations["label"] == 1.0)
            print("Label DME", n1)
            params["Val_Label DME"] = n1
            n2 = sum(val_annotations["label"] == 2.0)
            print("Label OV", n2)
            params["Val_Label OV"] = n2
            n3 = sum(val_annotations["label"] == 3.0)
            print("Label Normal", n3)
            params["Val_Label Normal"] = n3

            print("---------------------------Test---------------------------")        
            n1 = sum(test_annotations["label"] == 1.0)
            print("Label DME", n1)
            params["Test_Label DME"] = n1
            n2 = sum(test_annotations["label"] == 2.0)
            print("Label OV", n2)
            params["Test_Label OV"] = n2
            n3 = sum(test_annotations["label"] == 3.0)
            print("Label Normal", n3)
            params["Test_Label Normal"] = n3

            class_count = train_annotations.label.value_counts()
            class_weights = 1 /class_count
            train_samples_weight = np.array([class_weights[t] for t in (train_annotations.label.values)])

            mean, std = utils.check_mean_std(train_annotations)
            train_dataset = utils.Dataset(train_annotations, img_dir, mean, std, transform=resize_crop)
            train_sampler = WeightedRandomSampler(train_samples_weight, len(train_dataset), replacement=True)
                    
            val_dataset = utils.Dataset(val_annotations, img_dir, mean, std, transform=resize_crop)
                    
            train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=5, pin_memory=True, generator=g, sampler=train_sampler)  
            val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=5, pin_memory=True, generator=g)
                    
            if torch.cuda.is_available():
                model.to('cuda:0', non_blocking=True)
                    
            early_stop_callback = EarlyStopping(monitor="val_fb_score", min_delta=0.00, patience=pat, verbose=False, mode="max")

            trainer = Trainer(default_root_dir= model_path + "vgg19" +  "_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE) + ".ckpt", max_epochs=300, logger=neptune_logger, callbacks= [early_stop_callback], accelerator="gpu", precision="bf16-mixed", accumulate_grad_batches=1)
            tuner = Tuner(trainer)
                    
            optimizer = torch.optim.Adam(
                                params=model.parameters(), 
                                lr=learning_rate)
            optimizer.load_state_dict(checkpoint["optimizer_states"][0])
            lit_model = utils.LitModel(model, beta, BATCH_SIZE, learning_rate, "/home/vm/metrics/test/torch_lit/" + str(BATCH_SIZE) + "_fold_" + str(1))
                    
                    ##lr_finder = tuner.lr_find(model = lit_model, train_dataloaders=train_dataloader, val_dataloaders= val_dataloader, update_attr=True, max_lr = 0.0005, min_lr = 10e-6,  num_training=1000)
                    ##fig = lr_finder.plot(suggest=True)
                    ##fig.savefig("/home/vm/lr_suggest" + "_fold_" + str(fold)  + "_batch_size_" + str(BATCH_SIZE) +".pdf")
                    ##fig.clf()
                    ##fig.close("all")
                    #del model_copy
                    ##lit_model.learning_rate = lr_finder.suggestion()

                    ##params["learning_rate"] = lr_finder.suggestion()

            neptune_logger.log_hyperparams(params)

            torch.compile(lit_model)
            test_dataset = utils.Dataset(test_annotations, img_dir, mean, std, transform=resize_crop)
            test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), num_workers=5, pin_memory=True, generator=g)
            train_dataloader, val_dataloader, test_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader, test_dataloader)
            trainer.fit(lit_model, train_dataloader, val_dataloader)
                    
            trainer.test(model= lit_model, dataloaders= test_dataloader) #, ckpt_path="best")
            neptune_logger.experiment["torch_lit_cm"].upload("/home/vm/metrics/test/torch_lit/"+ str(BATCH_SIZE) + "_fold_" + str(1) + "_conf_matrix.png")
            neptune_logger.experiment["torch_lit_roc_curve"].upload("/home/vm/metrics/test/torch_lit/"+ str(BATCH_SIZE) + "_fold_" + str(1) + "_roc_curve.png")

            test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__(), pin_memory=True, generator=g)

            test_results_y_hat = trainer.predict(model= lit_model, dataloaders=test_dataloader) #, ckpt_path="best")
            test_results_y_hat_ = torch.cat([test_result for test_result in test_results_y_hat])
            m = nn.Softmax(dim=1)
            test_results_prob = m(test_results_y_hat_)
            test_results_pred = torch.argmax(test_results_y_hat_, dim=1)
            test_results_y = torch.cat([y for _, y in test_dataloader])

            #utils.plot_ROC_curve(test_results_y.cpu().detach().to(torch.float16).long(), test_results_prob.cpu().detach().to(torch.float16), metrics_path + "test/" + "roc_curve_" + "vgg19" +  "_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE))
            #utils.plot_confusion_matrix(test_results_y.cpu().detach().to(torch.float16).long(), test_results_pred.cpu().detach().to(torch.float16), metrics_path + "test/" + "cm_" + "vgg19" +  "_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE))
            #for i in range(num_classes):
            #            if i == 0:
            #               neptune_logger.experiment["test/ROC_DME"].upload(metrics_path + "test/roc_curve_vgg19_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE) + "class_" + str(i) + ".png")                
            #            if i == 1:
            #               neptune_logger.experiment["test/ROC_OV"].upload(metrics_path + "test/roc_curve_vgg19_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE) + "class_" + str(i) + ".png")
            #            if i == 2:
            #               neptune_logger.experiment["test/ROC_NORMAL"].upload(metrics_path + "test/roc_curve_vgg19_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE) + "class_" + str(i) + ".png")
                        #if i == 3:
                        #   neptune_logger.experiment["test/ROC_NORMAL"].upload(metrics_path + "test/roc_curve_vgg19_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE) + "class_" + str(i) + ".png")                

            #neptune_logger.experiment["test/cm"].upload(metrics_path + "test/cm_vgg19_fold_" + str(1)  + "_batch_size_" + str(BATCH_SIZE) + ".png")     
            precision, recall, fscore, fbeta_score_ = utils.classwise_precision_recall_accuracy(test_results_y.cpu().detach().to(torch.float16).long(), test_results_pred.cpu().detach().to(torch.float16).long())
            test_results_y_one_hot = F.one_hot(test_results_y, num_classes=3)
            test_results_pred_one_hot = F.one_hot(test_results_pred, num_classes=3)

            for lb in range(len(np.unique(test_results_y))):    
                        if lb ==0:
                            neptune_logger.experiment["test/DME_precision"].append(precision[lb])
                            neptune_logger.experiment["test/DME_recall"].append(recall[lb])
                            neptune_logger.experiment["test/DME_fscore"].append(fscore[lb])
                            neptune_logger.experiment["test/DME_fbeta_score"].append(fbeta_score_[lb])
                            neptune_logger.experiment["test/DME_accuracy"].append(accuracy_score(test_results_y_one_hot[:,lb], test_results_pred_one_hot[:,lb]))
                        if lb ==1:
                            neptune_logger.experiment["test/OV_precision"].append(precision[lb])
                            neptune_logger.experiment["test/OV_recall"].append(recall[lb])
                            neptune_logger.experiment["test/OV_fscore"].append(fscore[lb])
                            neptune_logger.experiment["test/OV_fbeta_score"].append(fbeta_score_[lb])
                            neptune_logger.experiment["test/OV_accuracy"].append(accuracy_score(test_results_y_one_hot[:,lb], test_results_pred_one_hot[:,lb]))
                        if lb ==2:
                            neptune_logger.experiment["test/Normal_precision"].append(precision[lb])
                            neptune_logger.experiment["test/Normal_recall"].append(recall[lb])
                            neptune_logger.experiment["test/Normal_fscore"].append(fscore[lb])
                            neptune_logger.experiment["test/Normal_fbeta_score"].append(fbeta_score_[lb])
                            neptune_logger.experiment["test/Normal_accuracy"].append(accuracy_score(test_results_y_one_hot[:,lb], test_results_pred_one_hot[:,lb]))
                        # if lb ==3:
                        #     neptune_logger.experiment["test/NORMAL_precision"].append(precision[lb])
                        #     neptune_logger.experiment["test/NORMAL_recall"].append(recall[lb])
                        #     neptune_logger.experiment["test/NORMAL_fscore"].append(fscore[lb])
                        #     neptune_logger.experiment["test/NORMAL_fbeta_score"].append(fbeta_score_[lb])
                        #     neptune_logger.experiment["test/NORMAL_accuracy"].append(accuracy_score(test_results_y_one_hot[:,lb], test_results_pred_one_hot[:,lb]))

            gl_precision, gl_recall, gl_fscore, gl_fbeta_score_ = utils.gobal_precision_recall_accuracy(test_results_y_one_hot, test_results_pred_one_hot)
            gl_accuracy = accuracy_score(test_results_y, test_results_pred)
            neptune_logger.experiment["test/gl_precison"].append(gl_precision)
            neptune_logger.experiment["test/gl_recall"].append(gl_recall)
            neptune_logger.experiment["test/gl_fscore"].append(gl_fscore)
            neptune_logger.experiment["test/gl_fbeta_score_"].append(gl_fbeta_score_)
            neptune_logger.experiment["test/gl_accuracy"].append(gl_accuracy)

            neptune_logger.experiment.stop()
            del model
                    #del best_model
            del train_dataloader
            del val_dataloader
            gc.collect()
            torch.cuda.empty_cache()