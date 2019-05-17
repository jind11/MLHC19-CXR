import os
import numpy as np
from PIL import Image
from skimage.external import tifffile 
import skimage 
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

#--------------------------------------------------------------------------------

vectorizer = TfidfVectorizer(analyzer='word', stop_words='english', max_features=9000)

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform, fold):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.listReportPaths = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")

        pathReportDirectory = pathImageDirectory.replace('tiffs_20k', 'reports_20k_tok')

        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()

                imageFileName = lineItems[0]
                imagePath = os.path.join(pathImageDirectory, imageFileName)
                reportFileName = imageFileName.split('.')[0] + '.txt'
                reportPath = os.path.join(pathReportDirectory, reportFileName)
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)
                self.listReportPaths.append(reportPath)
            
        fileDescriptor.close()

        reports = []
        for f in self.listReportPaths:
            reports.append(open(f, 'r').read())

        if fold == 'train':
            self.reports = vectorizer.fit_transform(reports).toarray()
        else:
            self.reports = vectorizer.transform(reports).toarray()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        reportPath = self.listReportPaths[index]
        
        #imageData = Image.open(imagePath).convert('RGB')
        imageData = tifffile.imread(imagePath)
        imageData = skimage.exposure.equalize_hist(imageData)
        imageData = (imageData*255).astype('uint8')
        imageData = Image.fromarray(imageData).convert('RGB')
        
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None: 
            imageData = self.transform(imageData)

        #print(type(imageData), type(imageLabel))
        return imageData, imageLabel, self.reports[index]
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 
    