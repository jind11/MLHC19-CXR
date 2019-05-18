In this code, we have three notebooks:
1) preprocessing.ipynb - this code was used for preprocessing the data from MIMIC-CXR.
This data is not publicly available, and thus has not been included here.

2) chexnet/chexnet.ipynb - this notebook was used for training and evaluating Chexnet.

3) chexnet/tienet.ipynb - this notebook was used for training and evaluating tienet.

Also included in chexnet/:
data4_weights.npy - initially Word2Vec trained word embeddings of the dataset
data_4_pubmed_embeddings - a gensim Word2Vec model pretrained on Pubmed literature and fine-tuned on the reports
pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin - the pretrained Word2Vec model on Pubmed literature 


dataset/*.txt - these are text files used for specifying the paths to the images and reports, as well as the labels for each image and report.



Please note that because the data is not included in this repository, these will not run properly.
