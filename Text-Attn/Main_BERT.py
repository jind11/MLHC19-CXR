import os
import numpy as np
import time
import sys
import argparse

from ChexnetTrainer_BERT import ChexnetTrainer

nnArchitecture = 'DENSE-NET-121'
nnIsTrained = True
nnClassCount = 14
imgtransResize_train = 224
imgtransResize_test = 256
imgtransCrop = 224


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="CXR Classifier")

    # main parameters
    parser.add_argument("--dump_dir", type=str, default="./models/BERT-2/",
                        help="Experiment dump path")
    parser.add_argument("--model", type=str, default="densenet121",
                        help="Model type")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("--batch_size_train", type=int, default=16,
                        help="Batch size for train and val")
    parser.add_argument("--batch_size_test", type=int, default=16,
                        help="Batch size for test")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="weight decay for optimization")
    parser.add_argument("--attn_size", type=int, default=100,
                        help="Attention vector size")
    parser.add_argument("--patience", type=int, default=5,
                        help="Patience to reduce learning rate after no improvement")

    # data
    parser.add_argument("--resume_path", type=str, default="",
                        help="Checkpoint path to resume from")
    parser.add_argument("--data_dir", type=str, default="../data",
                        help="directory for data")

    # for text modeling
    parser.add_argument("--bert_model", type=str,
                        default='/data/medg/misc/jindi/nlp/tools/BERT/clinical_pretrained/biobert_pretrain_output_all_notes_150000',
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--text_rep_size", type=int, default=768,
                        help="Text representation vector size")
    parser.add_argument("--lr_bert", type=float, default=2e-5,
                        help="learning rate for bert model training")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    return parser

#-------------------------------------------------------------------------------- 

def main ():
    parser = get_parser()
    params = parser.parse_args()

    timestampLaunch = runTrain(params)
    pathModel = params.dump_dir + 'm-' + timestampLaunch + '.pth.tar'
    runTest(params, pathModel)
  
#--------------------------------------------------------------------------------   

# Training code
def runTrain(params):

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # ---- Path to the directory with images
    #     pathDirDataTrain = [bdir('data/dataset_20k.npy'), bdir('data/labels_20k.npy')]
    #     pathDirDataVal = [bdir('data_2/dataset_20k_2.npy'), bdir('data_2/labels_20k_2.npy')]
    pathDirDataTrain = '../data/train/tiffs_20k'
    pathDirDataVal = '../data/test/tiffs_20k'
    #     pathDirDataVal = None

    # ---- Paths to the files with training, validation and testing sets.
    # ---- Each file should contains pairs [path to image, output vector]
    # ---- Example: images_011/00027736_001.png 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    pathFileTrain = '../data/train/train.txt'
    pathFileVal = '../data/test/val.txt'
    # pathFileTest = './dataset/test_1.txt'

    # pathModel = 'm-' + timestampLaunch + '.pth.tar'

    print ('Training NN architecture = ', nnArchitecture)
    ChexnetTrainer.train(pathDirDataTrain, pathDirDataVal, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained,
                         nnClassCount, params.batch_size_train, params.num_epochs, imgtransResize_train, imgtransCrop,
                         timestampLaunch, None, text_rep_size=params.text_rep_size, attn_size=params.attn_size,
                         dump_dir=params.dump_dir, bert_model=params.bert_model,
                         max_seq_length=params.max_seq_length)

    return timestampLaunch

#-------------------------------------------------------------------------------- 

def runTest(params, pathModel):
    # Test set
    pathDirData = '../data/test/tiffs_20k'
    pathFileTest = '../data/test/test.txt'

    timestampLaunch = ''
    print("Test set:")
    ChexnetTrainer.test(pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, params.batch_size_test,
                        imgtransResize_test, imgtransCrop, timestampLaunch, bert_model=params.bert_model,
                        max_seq_length=params.max_seq_length)

#-------------------------------------------------------------------------------- 

if __name__ == '__main__':
    main()





