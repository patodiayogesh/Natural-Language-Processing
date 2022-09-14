
# 1 Training PPMI-SVD and Word2Vec word embeddings

train_embeddings.py:
Contains Class to generate PPMI SVD based embeddings. Uses TruncatedSVD to compute the SVD.
Contains Class to generate Word2Vec based embeddings.
Variatons supported:
* Context (2,5,10)
* Dimension (50,100,300)
* Negative Sampling (1,15)


# 2 Contextualized word embeddings: BERT

bert_wic.py: Utilizes Bert tokenizer and transformer to generate the word embeddings.

models.py: File containing NN architecture and training functions

A Dense Neural Network is used for predicting the word context similarity. The DNN consists of 3 Linear layers and an activation ('relu') function between each layer. 
The target word embeddings of each pair are sent in batches of size 128 to the model for training with epoch =20. 
A validation set was created from the training set to optimize the loss function. 
* Adam Optimizer was used as optimizer with a learning rate of 0.02. 
* Layer 1 converts to dimension 128. 
* Layer 2 converts to dimension 64.
* Layer 3 gives the result class.

# 3 Probing BERT

explore_bert.py: Extracts target hidden state BERT representations from every layer of `bert-base-cased` for three evaluation corpora: the CONLL 2003 task, which contains sentences annotated with 1) part-of-speech and 2) named entity tags and SemEval 2010 Task 8 which contains sentences annotated with 3) entities and the relation expressed between those entities by the sentence.

Plot the F1 scores for each layer on:
* POS tagging (using CONLL)
* Named Entity Recognition(NER) (using CONLL)
* Relation Extraction(REL) (using SemEval)

![](https://github.com/patodiayogesh/Natural-Language-Processing/blob/main/SVD%2C%20Word2Vec%20and%20BERT%20word%20embeddings/3_explore_bert/plot.png)
