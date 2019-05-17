This folder contains the code for the Text-Guided Attention model.

To reproduce the baseline model--CheXNet model, please run the command line:

```
CUDA_VISIBLE_DEVICES=XXX python Main.py
```

To reproduce the text-guided attention model with TF-IDF text modeling, , please run the command line:

```
CUDA_VISIBLE_DEVICES=XXX python Main_TFIDF.py
```

To reproduce the text-guided attention model with BERT text modeling, , please run the command line:

```
CUDA_VISIBLE_DEVICES=XXX python Main_BERT.py
```
