import os
import numpy as np
from PIL import Image
from skimage.external import tifffile 
import skimage 
import torch
from torch.utils.data import Dataset
import gensim 
import re
#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform, report_exists=False):
        # if report_exists is True, then the report is on column 1 and needs to be taken into account. 
        if report_exists:
            starting_index_for_label = 2
        else:
            starting_index_for_label = 1
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        # labels = np.load(pathDatasetFile) 
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                imageLabel = lineItems[starting_index_for_label:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        #imageData = Image.open(imagePath).convert('RGB')
        imageData = tifffile.imread(imagePath)
        imageData = skimage.exposure.equalize_hist(imageData)
        imageData = (imageData*255).astype('uint8')
        imageData = Image.fromarray(imageData).convert('RGB')
        
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None: 
            imageData = self.transform(imageData)
        #print(type(imageData), type(imageLabel))
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #-------------------------------------------------------------------------------- 


##### The following are functions for processing the text and image data for TieNet.
#####

def tokenize_report(report_file):
    '''
    Convert one report to tokens.
    '''
    with open(report_file,'r') as f:
        g = f.read().replace('\n', '')
    f.close()
    tokens = clean_text(g)
    return tokens



def preprocess_all_reports(report_dir):
    '''
    Convert reports to tokens.
    '''
    filenames = os.listdir(report_dir)
    # Preprocess the report
    processed_reports = [] 
    for filename in filenames:
        tokens = tokenize_report(os.path.join(report_dir, filename))
        processed_reports.append(tokens) 
    return processed_reports


def clean_text(t):
    return re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').replace('\\', '').replace("'",'').strip().lower()).split()


def index_report(tokens, model, vocab_size, include_start_end=True):
    '''
    Turns tokens of report into array of the appropriate indices.

    vocab_size only matters if include_start_end is True 
    '''
    indices = [] 
    if include_start_end:
        # Make these indices to be at the end. Start is at vocab_size index, and 
        # end is at vocab_size-2 index. 
        indices.append(vocab_size-2) # to start it 
    for token in tokens:
        try:
            ind = model.vocab[token].index+1 # because 0 is the padding / unknown word token 
        except:
            ind = 0 # this takes care of OOV cases, they are treated like padding 
        indices.append(ind) 
    if include_start_end:
        indices.append(vocab_size-1)
    return indices 

def get_weight_matrix(model, padding=True, add_start_end_token=True): 
    # Make a weight matrix 
    # Padding will be at 0 index 
    weights_numpy = model.vectors 
    if padding:
        weights_numpy = np.concatenate((np.zeros((1,*weights_numpy.shape[1:])),weights_numpy),axis=0)
    weights = torch.FloatTensor(weights_numpy) 

    if add_start_end_token:
        eps = 1e-3 # the standard deviation of the small random vector we choose for start and end tokens 
        start_token = torch.FloatTensor(np.random.randn(1,weights.shape[1])*eps)
        end_token = torch.FloatTensor(np.random.randn(1,weights.shape[1])*eps)
        weights = torch.cat((weights,start_token,end_token))
    return weights


    
class TienetTrainDatasetGenerator(Dataset):
    
    #-------------------------------------------------------------------------------- 
    # For training, we load the report files 
    def __init__ (self, pathImageDirectory, pathReportDirectory, pathDatasetFile, 
                    embeddingModelFile, transform, maxReportLength):
    
        self.embeddingModel = gensim.models.KeyedVectors.load(embeddingModelFile)
        self.listImagePaths = []
        self.listReportPaths = [] 
        self.listImageLabels = []
        self.transform = transform
        self.maxReportLength = maxReportLength # pad to this amount 
    
        #---- Open file, get image paths and labels
    
        fileDescriptor = open(pathDatasetFile, "r")
        # labels = np.load(pathDatasetFile) 
        
        #---- get into the loop
        line = True
        
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
          
                lineItems = line.split()
                
                imagePath = os.path.join(pathImageDirectory, lineItems[0])
                reportPath = os.path.join(pathReportDirectory, lineItems[1])
                imageLabel = lineItems[2:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listReportPaths.append(reportPath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()
    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        #imageData = Image.open(imagePath).convert('RGB')
        imageData = tifffile.imread(imagePath)
        imageData = skimage.exposure.equalize_hist(imageData)
        imageData = (imageData*255).astype('uint8')
        imageData = Image.fromarray(imageData).convert('RGB')
        
        imageLabel= torch.FloatTensor(self.listImageLabels[index])
        if self.transform != None: 
            imageData = self.transform(imageData)


        # Get the reports
        reportPath = self.listReportPaths[index]
        reportText = tokenize_report(reportPath)

        # Using the preexisting word2vec model, we turn into indices
        # We are adding 3 to vocab to include <start> (len(vocab)-2), <end> (len(vocab)-1), and "unknown" (0)
        textData = index_report(reportText, self.embeddingModel, len(self.embeddingModel.vocab)+3,
                                include_start_end = True)
        textData = torch.LongTensor(textData) 

        # Also get the number of actual words before padding
        textLen = textData.size(0)

        # Now pad to the max sequence length of the data 
        temp_pad_layer = torch.nn.ConstantPad1d((0,self.maxReportLength-len(textData)), 0)
        textData = temp_pad_layer(textData)
        #print(imageData.shape, textData.shape, textLen) 
         

        #print(type(imageData), type(imageLabel))
        return imageData, textData, textLen, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
