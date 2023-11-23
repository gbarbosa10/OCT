# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 09:16:03 2023

@author: guiti
"""
from torch.utils.data import Dataset
import numpy as np
import cv2
from torchvision.io import read_image
import torch
import os 
import torch
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes
import pytorch_lightning as pl
import lightning as L
from torch.utils.data import DataLoader, Dataset
#from nvidia.dali.pipeline import pipeline_def
#import nvidia.dali.types as types
#import nvidia.dali.fn as fn
#from nvidia.dali.plugin.pytorch import DALIGenericIterator

#@pipeline_def(num_threads=4, device_id=0)
#def get_dali_pipeline(images_dir):
#    images, labels = fn.readers.file(
#        file_root=images_dir, random_shuffle=True, name="Reader")
    # the rest of processing happens on the GPU as well
#    images = fn.resize(images, resize_x=256, resize_y=256)
#    return images, labels


#train_data = DALIGenericIterator(
#    [get_dali_pipeline(batch_size=16)],
#    ['data', 'label'],
#    reader_name='Reader'
#)


#class MNISTDataModule(pl.LightningDataModule):
#    def __init__(self, data_dir: str = "./"):
#        super().__init__()
#        self.data_dir = data_dir
#        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#    def prepare_data(self):
        # download
#        MNIST(self.data_dir, train=True, download=True)
#        MNIST(self.data_dir, train=False, download=True)

#    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
#        if stage == "fit" or stage is None:
#            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
#            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        # Assign test dataset for use in dataloader(s)
#        if stage == "test" or stage is None:
#            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

#        if stage == "predict" or stage is None:
#            self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)

#    def train_dataloader(self):
#        return DataLoader(self.mnist_train, batch_size=32)

#    def val_dataloader(self):
#        return DataLoader(self.mnist_val, batch_size=32)

#    def test_dataloader(self):
#        return DataLoader(self.mnist_test, batch_size=32)

#    def predict_dataloader(self):
#        return DataLoader(self.mnist_predict, batch_size=32)


class Dataset(Dataset):
   #Define class Dataset: -------------------------------------------------------------------------------------
    # args: annotation_file, img_dir, threshold, transform, filt, crop
    # annotations_file: csv with data input information (filepath and label)
    # img_dir: image directory for the current images available 
    # transform: applies a data augmentations operation to the input of the network dwfault = None
    # threshold: pixel value (0-255) that determines the pixels turned to black in the preprocessing 1 function 
    # filt: filter size for top-hat application in preprocessing2
    # crop: 0 - there is no crop, 1 - crop top of the image, 2 - crop top and bottom of the image, 

   def __init__(self, annotations_file, img_dir, mean, std, transform=None):  
       self.img_labels = annotations_file
       self.transform = transform
       self.img_dir = img_dir
       self.mean = mean
       self.std = std
       
   def __len__(self):
       return len(self.img_labels)

   def __getitem__(self, idx):
       mean_ = self.mean
       std_ = self.std

       img_path = self.img_labels["filename"].iloc[idx]
       
       image = read_image( self.img_dir + img_path)
          
       label = np.array(self.img_labels["label"].iloc[idx])

       image = image/255    
       image1 = (image - mean_) / ( std_)
       label = label - 1
       if image.shape[0] == 1:
            image = torch.concatenate((image1, image1), axis=0)
            image = torch.concatenate((image, image1), axis=0)
       if self.transform:
            image = self.transform(image)
       return image, int(label)

def preprocessing1(img, th):
    
        # Convert to PIL Image
        img = read_image(img)
        # Resize image
        im_size = img.size()
       
        # Preprocessing 1 #####################################
        thresh_value = th
        # Select the top and bottom regions of the image
        top_region = img[:, :int(im_size[1]*0.6),:]
        
        bottom_region = img[:, im_size[1] - int(im_size[1]*0.3):,:]
        
        #Select left and right sides of the image
        
        left_region = img[:, :, :int(im_size[2]*0.05)]
        right_region = img[:, :,im_size[2] - int(im_size[2]*0.05):]
        
        # Apply thresholding to the top, bottom, left and right regions
        top_region_thresh = np.where(top_region > thresh_value, 0, top_region)
        bottom_region_thresh = np.where(bottom_region > thresh_value, 0, bottom_region)
        
        left_region_thresh = np.where(left_region > thresh_value, 0, left_region)
        right_region_thresh = np.where(right_region > thresh_value, 0, right_region)
        
        
        # Combine the thresholded top and bottom regions with the original image
        img_arr = np.array(img)
        img_arr[:, :int(im_size[1]*0.6),:] = top_region_thresh
        img_arr[:, im_size[1] - int(im_size[1]*0.3):,:] = bottom_region_thresh
        img_arr[:, :,:int(im_size[2]*0.05)] = left_region_thresh
        img_arr[:, :,im_size[2] - int(im_size[2]*0.05):] = right_region_thresh
        
        #img_arr = cv2.fastNlMeansDenoising(img_arr, None, h=7, templateWindowSize=5, searchWindowSize=21)
        #img_arr=img
        #img_arr = cv2.resize(img_arr, (224, 224))
        img_arr = torch.from_numpy(img_arr)
        return img_arr

 
def top_hat(image, filter_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                               filter_size)
     
    tophat_img = cv2.morphologyEx(image, 
                          cv2.MORPH_TOPHAT,
                          kernel)
    
    clean_im = np.array(image) - tophat_img
    return clean_im

def select_contours(image, filter_size):

    if filter_size:
        top_hat_im = top_hat(image, filter_size)
    else:
        top_hat_im = image
        
    ret,th1 = cv2.threshold(top_hat_im, 10,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    whole_im = binary_fill_holes(th1)
    kernel = np.ones((5, 5), np.uint8)
    dil_image = cv2.dilate(np.array(whole_im*255, np.uint8), kernel, iterations=1)
    
    contours, hierarchy = cv2.findContours(np.array(dil_image*255, np.uint8), 
        cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    return contours

def process_contours(contours):
    
    length = []
    for i in range(len(contours)):
        length.append(len(contours[i]))
    i_max = np.argsort(length)[-1]
    i_max_2 = False
    if len(length) > 1: 
        i_max_2 = np.argsort(length)[-2]
    
    max_cont = contours[i_max]
    x = max_cont[:,:, 0]
    y = max_cont[:,:, 1]
    xmin = min(x)[0]
    xmax = max(x)[0]
    if i_max_2:
        if len(contours[i_max_2]) > 900:
            max_cont_2 = contours[i_max_2]
            x2 = max_cont_2[:,:, 0]
            y2 = max_cont_2[:,:, 1]
            y2_min = min(y2)[0]
            ymin = min(y)[0]
            xmin2 = min(x2)[0]
            xmax2 = max(x2)[0]
            
            
            if y2_min < ymin:
                p1 = np.argmin(np.where(x2==xmin2, y2, 1000))
                p4 = np.argmin(np.where(x2==xmax2, y2, 1000))
                
                p2 = np.argmax(np.where(x==xmin, y, 0))
                p3 = np.argmax(np.where(x==xmax, y, 0))
                new_counts = tuple((max_cont[p2:p3, :, :], np.concatenate((max_cont_2[p4:, :, :], max_cont_2[:p1, :, :]), axis=0)))
                
                ymin = y2_min
                ymax = max(y)[0]
            else:
                p1 = np.argmin(np.where(x==xmin, y, 1000))
                p4 = np.argmin(np.where(x==xmax, y, 1000))
                
                p2 = np.argmax(np.where(x2==xmin2, y2, 0))
                p3 = np.argmax(np.where(x2==xmax2, y2, 0))
                new_counts = tuple((max_cont_2[p2:p3, :, :], np.concatenate((max_cont[p4:, :, :], max_cont[:p1, :, :]), axis=0)))
                
                ymin = min(y)[0]
                ymax = max(y2)[0]
        else:
       
            p1 = np.argmin(np.where(x==xmin, y, 1000))
            p4 = np.argmin(np.where(x==xmax, y, 1000))
       
            p2 = np.argmax(np.where(x==xmin, y, 0))
            p3 = np.argmax(np.where(x==xmax, y, 0))
            new_counts = tuple((max_cont[p2:p3, :, :], np.concatenate((max_cont[p4:, :, :], max_cont[:p1, :, :]), axis=0)))
       
            ymin = min(y)[0]
            ymax = max(y)[0]        
    else:
       
       p1 = np.argmin(np.where(x==xmin, y, 1000))
       p4 = np.argmin(np.where(x==xmax, y, 1000))
       
       p2 = np.argmax(np.where(x==xmin, y, 0))
       p3 = np.argmax(np.where(x==xmax, y, 0))
       new_counts = tuple((max_cont[p2:p3, :, :], np.concatenate((max_cont[p4:, :, :], max_cont[:p1, :, :]), axis=0)))
       
       ymin = min(y)[0]
       ymax = max(y)[0]
    return new_counts, ymin, ymax

def check_width(contours, im_size, threshold):
   
   check_flag = False 

   for contour in contours:
       x = contour[:,:, 0]
       if x!=[]:
            xmin = min(x)[0]
            xmax = max(x)[0]   
        
            if (xmin < im_size[1]*threshold or xmax > im_size[1]*(1 - threshold)):
                check_flag = True
    
   return check_flag
    
def preprocessing2(img, top_off, bottom_off, top_hat_, th):
        
        img_arr = preprocessing1(img, th)
        im_size = img_arr.size()
        img_arr = cv2.convertScaleAbs(np.array(img_arr[0,:,:]), 1.5, 2)
        
        img_og = read_image(img)
        
        #filter_size_list = [(5, 5), (3,3), (3, 1)]
        
        #for filt_size in filter_size_list: 
        contours = select_contours(img_arr, (5, 5))
        new_counts_, ymin, ymax = process_contours(contours)
            
        if check_width(new_counts_, im_size, 0.1) == False:
            contours = select_contours(img_arr, (3, 3))
            new_counts_, ymin, ymax = process_contours(contours)
            #break
            if check_width(new_counts_, im_size, 0.1) == False:
                contours = select_contours(img_arr, (3, 1))
                new_counts_, ymin, ymax = process_contours(contours)


        img = np.array(img_og[0,:,:], np.uint8)
        left_region = img[:,:int(im_size[2]*0.05)]
        right_region = img[:,im_size[2] - int(im_size[2]*0.05):]
        
        # Apply thresholding to the left and right regions
        left_region_thresh = np.where(left_region > 250, 0, left_region)
        right_region_thresh = np.where(right_region > 250, 0, right_region)
        
        img[:,:int(im_size[2]*0.05)] = left_region_thresh
        img[:,im_size[2] - int(im_size[2]*0.05):] = right_region_thresh
        # Apply Top-Hat
        if top_hat_:
            new_clean_im = top_hat(img, top_hat_)
        
        else:
            new_clean_im = img_arr.copy() 
        
        # Contorno Superior - Limpeza de Imagem
        if top_off==True:
            sup_count = new_counts_[1]
            x_sup = sup_count[:, :, 0]
            y_sup = sup_count[:, :,1]

            new_y_sup = []
            for x1 in np.unique(x_sup): # Retirar apenas o maior y para cada x
                    new_y_sup = np.concatenate((new_y_sup, min(np.where(x_sup==x1, y_sup, 1000))))
            
            for i in range(len(new_y_sup)):
                y_lim = new_y_sup[i]
                new_clean_im[:int(y_lim), np.unique(x_sup)[i]] = 0
        else:
            top_region = img[:int(im_size[1]*0.3),:]
            top_region_thresh = np.where(top_region > 210, 0, top_region)
            img[:int(im_size[1]*0.3),:] = top_region_thresh

        # Contorno Inferior - Limpeza da Imagem
        if bottom_off == True:
            inf_count = new_counts_[0]
            x_inf = inf_count[:, :, 0]
            y_inf = inf_count[:, :,1]

            new_y_inf = []
            for x2 in np.unique(x_inf): # Retirar apenas o menor y para cada x
                new_y_inf = np.concatenate((new_y_inf, max(np.where(x_inf==x2, y_inf, 0))))

            for i in range(len(new_y_inf)):
                y_lim = new_y_inf[i]
                new_clean_im[int(y_lim):, np.unique(x_inf)[i]] = 0
        else:
            bottom_region = img[im_size[1] - int(im_size[1]*0.3):,:]
            bottom_region_thresh = np.where(bottom_region > 240, 0, bottom_region)
            img[im_size[1] - int(im_size[1]*0.3):,:] = bottom_region_thresh
            
        new_clean_im = torch.from_numpy(new_clean_im[ymin:ymax,:])
        new_clean_im = torch.unsqueeze(new_clean_im, 0)
        return new_clean_im

def check_mean_std(csv):
    train_mean = 0
    train_std = 0

    for line in csv["filename"]: 
        image = read_image("/home/vm/all_images/" + line)
        train_mean += image.float().mean().item()
        train_std += image.float().std().item()

    train_mean = train_mean / len(csv)
    train_std = train_std / len(csv)
    return train_mean, train_std


class CustomDataModule(L.LightningDataModule):
        def __init__(self, train_annotations, val_annotations, test_annotations, img_dir, mean, std,  batch_size=64, transform=None):
            super().__init__()
            self.train_annotations = train_annotations
            self.val_annotations = val_annotations
            self.test_annotations = test_annotations
            self.img_dir = img_dir
            self.mean = mean
            self.std = std
            self.transforms = transform

        def prepare_data(self):
            # download
            pass

        def setup(self, stage: str):

            #X, y = make_classification(
            #n_samples=20000,
            #n_features=100,
            #n_informative=10,
            #n_redundant=40,
            #n_repeated=25,
            #n_clusters_per_class=5,
            #flip_y=0.05,
            #class_sep=0.5,
            #random_state=123,
            #)

            #X_train, X_test, y_train, y_test = train_test_split(
            #X, y, test_size=0.2, random_state=1, shuffle=True
            #)

            #X_train, X_val, y_train, y_val = train_test_split(
            #X_train, y_train, test_size=0.1, random_state=1, shuffle=True
            #)

            self.train_dataset = Dataset(
            self.train_annotations, self.img_dir, self.mean, self.std
            , transform= self.transforms)

            self.val_dataset = Dataset(
            self.val_annotations, self.img_dir, self.mean, self.std
            , transform= self.transforms)

            self.test_dataset = Dataset(
            self.test_annotations, self.img_dir, self.mean, self.std
            , transform= self.transforms)

        def train_dataloader(self):
            train_loader = DataLoader(
                    dataset=self.train_dataset,
                    batch_size=32,
                    shuffle=True,
                    drop_last=True,
                    num_workers=0,
                    )
            return train_loader

        def val_dataloader(self):
            val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            )
            return val_loader

        def test_dataloader(self):
            test_loader = DataLoader(
            dataset=self.test_dataset, batch_size=32, shuffle=False, num_workers=0
            )
            return test_loader