import os
import numpy as np
from PIL import Image
from skimage.external import tifffile 
import skimage 
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer

#--------------------------------------------------------------------------------

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    def __init__ (self, pathImageDirectory, pathDatasetFile, transform, tokenizer, max_seq_length=128):
    
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
            with open(f, 'r') as file:
                report = file.read()
            reports.append(report)

        self.reports = convert_text_to_features(reports, max_seq_length, tokenizer)
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

        report_ids, report_mask, segment_ids = self.reports[index]
        return imageData, imageLabel, torch.LongTensor(report_ids), \
               torch.LongTensor(report_mask), torch.LongTensor(segment_ids)
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self.listImagePaths)
    
 #--------------------------------------------------------------------------------

def convert_text_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = {}
    for (ex_index, example) in enumerate(examples):
        tokens = tokenizer.tokenize(example)

        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[-(max_seq_length - 2):]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features[ex_index] = (input_ids, input_mask, segment_ids)

    return features