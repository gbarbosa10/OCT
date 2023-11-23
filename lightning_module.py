import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import MulticlassRecall
from torchmetrics.classification import MulticlassPrecision
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassFBetaScore
from torchmetrics.classification import MulticlassROC
from torchmetrics.classification import MulticlassConfusionMatrix
from torchmetrics import Accuracy
import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import lightning as L

class LitModel(pl.LightningModule):
    def __init__(self, encoder, beta, batch_size, learning_rate, save_path, num_class): #optimizer):
        super().__init__()
        self.model = encoder
        
        self.beta = beta
        self.batch_size = batch_size
        #self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.auroc = MulticlassAUROC(num_classes=num_class)
        self.roc = MulticlassROC(num_classes=num_class)
        self.acc = MulticlassAccuracy(num_classes=num_class)
        self.recall = MulticlassRecall(num_classes=num_class)
        self.precision = MulticlassPrecision(num_classes=num_class)
        self.f1score = MulticlassF1Score(num_classes=num_class)
        self.fbetascore = MulticlassFBetaScore(num_classes=num_class, beta=self.beta)
        self.cm = MulticlassConfusionMatrix(num_classes=num_class)
        self.save_path = save_path

        self.total_pred = torch.tensor([]).cpu()
        self.total_y = torch.tensor([]).cpu()
        self.total_prob = torch.tensor([]).cpu()


    def training_step(self, batch, batch_idx):
        x, y = batch
        
        y = y.type(torch.LongTensor).cuda()
        tin = time.time()
        y_hat = self.model(x)
        tend = time.time()
        time_ = tend - tin
        loss = F.cross_entropy(y_hat, y)
        probs = torch.softmax(y_hat, dim =1)
        #pred = torch.argmax(y_hat, dim=1)
        self.log("train_duration", time_, on_step = False, on_epoch = True)

        self.log("train_fb_score", self.fbetascore(probs, y), on_step = False, on_epoch = True)
        self.log("train_loss", loss, on_step = False, on_epoch = True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        y = y.type(torch.LongTensor).cuda()
        tin = time.time()
        y_hat = self.model(x)
        tend = time.time()
        time_ = tend - tin

        val_loss = F.cross_entropy(y_hat, y)
        
        probs = torch.softmax(y_hat, dim =1)
        #pred = torch.argmax(y_hat, dim=1)
        self.log("val_duration", time_, on_step = False, on_epoch = True)
        self.log("val_fb_score", self.fbetascore(probs, y), on_step = False, on_epoch = True)
        self.log("val_loss", val_loss, on_step = False, on_epoch = True)
        
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.LongTensor).cuda()
        tin = time.time()
        y_hat = self.model(x)
        tend = time.time()
        time_ = tend - tin
        test_loss = F.cross_entropy(y_hat, y)
        probs = torch.softmax(y_hat, dim =1)

        self.log("test_duration", time_, on_step = False, on_epoch = True)

        self.log("test_auc", self.auroc(probs, y) , on_step = False, on_epoch = True)
        self.log("test_precision", self.precision(probs, y), on_step = False, on_epoch = True)
        self.log("test_recall", self.recall(probs, y) , on_step = False, on_epoch = True)
        self.log("test_fb_score", self.fbetascore(probs, y) , on_step = False, on_epoch = True)
        self.log("test_acc", self.acc(probs, y), on_step = False, on_epoch = True)
        self.log("test_loss", test_loss, on_step = False, on_epoch = True)

        self.roc.update(probs, y)
        fig_, ax_ = self.roc.plot(score=True)
        fig_.savefig(self.save_path + "_roc_curve.png")
        fig_.clf()

        self.cm.update(probs, y)
        fig_, ax_ = self.cm.plot()
        fig_.savefig(self.save_path + "_conf_matrix.png")
        fig_.clf()

        return test_loss

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(
                                params=self.model.parameters(), 
                                lr=self.learning_rate)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #    self.optimizer, "min", factor =0.1,
        #    patience= self.patience,
        #    threshold = 1e-3
        #    )
        
        #return [self.optimizer], [scheduler]
        
        #return {
        #   'optimizer': self.optimizer,
        #   'lr_scheduler': scheduler, # Changed scheduler to lr_scheduler
        #   'monitor': 'val_loss'
        #}
        return optimizer

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.model(x)
        #pred = torch.argmax(y_hat, dim=1)
        #probs = torch.softmax(y_hat, dim =1)
        return y_hat
    
    def predictProbs_step(self, batch, batch_idx):
        x, _ = batch
        y_hat = self.model(x)
        probs = torch.softmax(y_hat, dim =1)
        return probs
    
    def predictY_step(self, batch, batch_idx):
        _, y = batch
        return y