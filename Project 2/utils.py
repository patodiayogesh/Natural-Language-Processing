"""
Emotion Classification with Neural Networks - Utils File
"""

# Imports
import nltk
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import math

# Global definitions - data
EMBEDS_FN = 'resources/glove.twitter.27B.100d.txt'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]


class EmotionDataset(Dataset):
    def __init__(self, data, word2idx=None, encoder=None, glove=None):
        """
        Dataset class to help load and handle the data, specific to our format.
        Provides X, y pairs.
        Create the dataset by providing the data as well as possible predefined preprocessing
        (e.g., to prevent data leakage between train and test).

        Args:
            :param data (pandas dataframe): data to represent
            :param word2idx: either the prepared {word: index} dictionary or None (to create one from the data)
            :param encoder: either a prepared label encoder or None (to create one from the data)
            :param glove: a prepared {word: embedding} dictionary, if creating the vocabulary
        """
        if word2idx is None and glove is None:
            raise ValueError("Must pass either a predefined vocabulary or a GloVe dictionary to make a dataset.")

        self.label_enc = encoder  # encoder may be None, in which case make_vectors() will handle it
        self.emotion_frame = data
        self.word2idx = word2idx if word2idx is not None else get_word2idx(get_tokens(data), glove)
        self.X, self.y, self.label_enc = make_vectors(self.emotion_frame, self.word2idx, self.label_enc)

    def __len__(self):
        return len(self.emotion_frame)

    def __getitem__(self, idx):
        X = self.X[:, idx]
        y = self.y[idx]

        return X, y


def get_word2idx(data, glove):
    """
    Calculate the {word: index} dictionary given a dataset and embeddings.
    Here, remove words based on whether they are in pretrained GloVe
    :param data: a list of tokenized data points (each data point should be a list of strings)
    :param glove: {word: embedding} dictionary, or just a list of words in vocabulary
    :return: word2idx, a {word: index} dictionary where <PAD> is 0 and <UNK> is the last word.
    """
    word2idx = {'<pad>': 0}
    for datapoint in data:
        for word in datapoint:
            if word in glove and word not in word2idx:
                word2idx[word] = len(word2idx)
    word2idx['<unk>'] = len(word2idx)

    return word2idx


def make_vectors(data, word2idx, label_enc=None):
    """
    Helper function: given text data and labels, transform them into vectors, where text becomes lists of word indices.
    Words not in the provided vocabulary get mapped to an "unknown" token, <UNK>.
    :param data: a pandas DataFrame including 'content' and 'sentiment' columns
    :param word2idx: a {word: index} dictionary defining the vocabulary for this data.
    :param label_enc: a OneHotEncoder that turns labels into one-hot vectors. Pass None to fit a new one from the data.
    :return: X (a list of lists of word indices), y (a numpy matrix of class indices), label_enc (as in parameters)
    """
    X = nn.utils.rnn.pad_sequence([torch.tensor([word2idx[word] if word in word2idx else word2idx['<unk>'] for word in
                                                 datapoint]) for datapoint in get_tokens(data)])
    y = data['sentiment'].to_numpy()
    if label_enc is None:
        label_enc = LabelEncoder()
        y = label_enc.fit_transform(y)
    else:
        y = label_enc.transform(y)
    return X, y, label_enc


def get_tokens(data):
    """
    Helper function to get the tokens in a whole dataset.
    :param data: a pandas DataFrame where the text is in a column named 'content'
    :return: a list of lists of words in the whole dataset (i.e., a list of nltk-tokenized data points)
    """
    return [nltk.word_tokenize(datapoint.lower()) for datapoint in data['content'].tolist()]


def get_data(data_fn):
    """
    Load the data from file and split it randomly into train, dev, and test sets.
    :param data_fn: the name of the .csv file in which the data is stored
    :return: train, dev, test; splits of the pandas dataframe
    """

    # Load the data
    df = pd.read_csv(data_fn)

    # Split the data into train, dev, and test
    # .sample() shuffles the data - set the random seed so we always grab the same data split(!)
    train, dev, test = np.split(df.sample(frac=1, random_state=4705), [int(.8 * len(df)), int(.9 * len(df))])

    # Grab some samples from the train set to view
    print("Random samples from train set")
    print(train.sample(10))
    print("\n\n")

    return train, dev, test


def vectorize_data(train, dev, test, batch_size, embedding_dim):
    """
    Transform the data from pandas DataFrame form into numerical data packed in DataLoaders.
    (Also create the pretrained embedding matrix in the proper order, which will be used in models).
    :param train: The pandas DataFrame corresponding to the training set
    :param dev: The pandas DataFrame corresponding to the development set
    :param test: The pandas DataFrame corresponding to the testing set
    :param batch_size: The batch size for all  datasets
    :param embedding_dim: The dimensionality of embeddings
    :return: train_generator, dev_generator, test_generator (DataLoaders for each dataset),
             embeddings (the matrix of embeddings in the same order as our vocabulary),
             train_data (the Dataset corresponding to the train set, including the word2idx dictionary and label encoder
    """

    # Fit an encoder for turning emotion names into class indices
    label_enc = LabelEncoder()
    label_enc.fit(np.array(LABEL_NAMES))
    print("Labels are encoded in the following class order:")
    print(label_enc.inverse_transform(np.arange(4)))
    print("\n\n")

    # Load the pretrained embeddings
    df = pd.read_csv(EMBEDS_FN, sep=" ", quoting=3, header=None, index_col=0)
    glove = {key: val.values for key, val in df.T.items()}

    # Create the three Datasets
    train_data = EmotionDataset(train, glove=glove, encoder=label_enc)
    dev_data = EmotionDataset(dev, word2idx=train_data.word2idx, encoder=label_enc)
    test_data = EmotionDataset(test, word2idx=train_data.word2idx, encoder=label_enc)

    print("Train:", len(train_data), "\nDev:", len(dev_data), "\nTest:", len(test_data))
    print("\n\n")

    # And now the three DataLoaders
    train_generator = DataLoader(dataset=train_data, batch_size=batch_size)
    dev_generator = DataLoader(dataset=dev_data, batch_size=batch_size)
    test_generator = DataLoader(dataset=test_data, batch_size=batch_size)

    # Create the embedding matrix for use in our models
    embeddings = np.zeros((len(train_data.word2idx), embedding_dim))  # vocab_length x embedding_dimension
    for word, i in train_data.word2idx.items():
        if word == '<pad>':
            pass  # Leave the "pad" embedding all zeroes
        elif word == '<unk>':
            embeddings[i] = glove['unk']  # GloVe has an embedding for 'unk'
        else:
            embeddings[i] = glove[word]  # Take all other words from GloVe

    # Convert embeddings to a PyTorch tensor
    embeddings = torch.tensor(embeddings)

    # Pass back train_data for access to word2idx and label_enc.
    return train_generator, dev_generator, test_generator, embeddings, train_data


class Experimental_LR_Scheduler:
    """
    Class Implementing Exponential Learning Scheduler

    Attributes:
    -----------
    gamma: Set decay rate
    """

    def __init__(self, gamma):
        self.gamma = gamma

    def update_learing_rate(self, optimizer, epoch_iteration_number,
                            loss, development_loss):
        """
        Function to change the learning rate if previous loss
        is equal or less than current loss on the dev set

        :param optimizer: optimizer object
        :param epoch_iteration_number: the iteration number during training
        :param loss: the loss calculated on the dev set
        :param development_loss: loss calculated on dev set in previous iteration
        :return: None
        """

        # Reduces the learning rate exponentially for the condition below
        if development_loss < loss:
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * math.exp(-self.gamma * \
                                                     epoch_iteration_number)


        return
