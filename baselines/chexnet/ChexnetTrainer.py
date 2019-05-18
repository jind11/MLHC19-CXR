import os
import numpy as np
import time
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func
from torch.utils.data import Dataset, DataLoader
from torch.nn.modules.loss import _Loss

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator

#-------------------------------------------------------------------------------- 

# class masked_BCE(y_pred, y_true):
#     return torch.nn.BCELoss(y_pred[y_true >= 0], y_true[y_true >= 0])   

class masked_BCE(_Loss):

    def __init__(self):
        super(masked_BCE, self).__init__()

    def forward(self, y_pred, y_true, y_word_pred, y_word_true):
        # loss = torch.nn.BCELoss()
        loss = torch.nn.BCELoss()
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        return loss(y_pred[y_true >= 0], y_true[y_true >= 0])


class TieNetLoss(_Loss):

    def __init__(self, alpha, beta_P=[1.0], beta_N=1.0, lambda_m=1.0):
        super(TieNetLoss, self).__init__()
        # Alpha is how much to weight one part of the loss functino over other 
        self.alpha = alpha 
        self.pos_weights = torch.FloatTensor(beta_P)
        self.lambda_m = torch.FloatTensor(lambda_m).cuda()

    def forward(self, y_pred, y_true, y_word_pred, y_word_true):
        # loss = torch.nn.BCELoss()
        #y_pred = y_pred.view(-1)
        #y_true = y_true.view(-1)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weights, reduction='none').cuda()
        p1 = self.lambda_m* loss(y_pred, y_true.cuda()) # classification loss 
        p1 = p1[y_true >= 0].mean()

        word_loss = torch.nn.CrossEntropyLoss()
        p2 = word_loss(y_word_pred, y_word_true)


        return self.alpha*p1 + (1-self.alpha)*p2

class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    

    

    def train (pathDirDataTrain, pathDirDataVal, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirDataTrain, pathDatasetFile=pathFileTrain, transform=transformSequence)
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)

        if pathDirDataVal != None:
            datasetVal =   DatasetGenerator(pathImageDirectory=pathDirDataVal, pathDatasetFile=pathFileVal, transform=transformSequence)
            dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
            
        
        # dataLoaderTrain = torch.utils.data.DataLoader(GetDataset(in_data_file = pathDirDataTrain[0],
        #     out_data_file = pathDirDataTrain[1]), batch_size = trBatchSize, shuffle=True)
        # dataLoaderVal = torch.utils.data.DataLoader(GetDataset(in_data_file = pathDirDataVal[0],
        #     out_data_file = pathDirDataVal[1]), batch_size = trBatchSize, shuffle=False)

        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        #loss = torch.nn.BCELoss(size_average = True)
        loss = masked_BCE()
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            lossTrain, losstensor = ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)

            if pathDirDataVal != None:
                lossVal, losstensor = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.item())
            
            if pathDirDataVal != None:
                if lossVal < lossMIN:
                    lossMIN = lossVal    
                    torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'models/m-' + launchTimestamp + '.pth.tar')
                    print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] val_loss= ' + str(lossVal) + ', train_loss= ' + str(lossTrain))
                else:
                    print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] val_loss= ' + str(lossVal) + ', train_loss= ' + str(lossTrain))
            else:
                if lossTrain < lossMIN:
                    lossMIN = lossTrain
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'models/m-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] train_loss= ' + str(lossTrain))
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        for batchID, (data, target) in tqdm(enumerate(dataLoader)):
                        
            target = target.cuda(async = True)
            # 
                 
            varInput = torch.autograd.Variable(data)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            lossvalue = loss(varOutput, varTarget)
            losstensorMean += lossvalue 
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            # look at the training loss as well 
            lossVal += lossvalue.item() 
            lossValNorm += 1

        outloss = lossVal / lossValNorm 
        losstensorMean = losstensorMean / lossValNorm 
        return outloss, losstensorMean
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate (dataLoader)):
                
                target = target.cuda(async=True)
                     
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)    
                varOutput = model(varInput)
                
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor
                
                #lossVal += losstensor.data[0]
                lossVal += losstensor.item()
                lossValNorm += 1
                
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm
        
        #losstensorMean = None
        return outLoss, losstensorMean
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().detach().numpy()
        datanpPRED = dataPRED.cpu().detach().numpy()


        total_num_samples = [] 
        for i in range(classCount):
            datanpGT_temp = datanpGT[:,i] 
            datanpGT_nonneg = datanpGT_temp[datanpGT_temp >= 0]
            datanpPRED_nonneg = datanpPRED[:,i]
            datanpPRED_nonneg = datanpPRED_nonneg[datanpGT_temp >= 0]
            num_samples = len(datanpGT_nonneg)
            outAUROC.append(roc_auc_score(datanpGT_nonneg, datanpPRED_nonneg))
            total_num_samples.append(num_samples)
            #print(outAUROC[i], num_samples)

        overallAUROC = np.sum(np.array(outAUROC)*np.array(total_num_samples))/np.sum(total_num_samples)

        outAUROC.append(overallAUROC)
        
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        # CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        #         'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        CLASS_NAMES = ['No Finding','Enlarged Cardiomediastinum','Cardiomegaly','Airspace Opacity','Lung Lesion','Edema',
                       'Consolidation','Pneumonia','Atelectasis','Pneumothorax','Pleural Effusion','Pleural Other','Fracture','Support Devices']

        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformList.append(normalize)
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = input.size()
                
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
                
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                
                outPRED = torch.cat((outPRED, outMean.data), 0)

            aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
            #aurocMean = np.array(aurocIndividual).mean()
            
            
            
            for i in range (0, len(aurocIndividual)-1):
                print (CLASS_NAMES[i], ' ', aurocIndividual[i])
            print ('AUROC mean ', aurocIndividual[-1])
     
        return model
#-------------------------------------------------------------------------------- 





