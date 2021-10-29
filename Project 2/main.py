"""
Emotion Classification with Neural Networks - Main File

YOGESH PATODIA
YP2607
"""

# Imports
import nltk
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
import copy

# Imports files
import utils
import models
import argparse
from utils import Experimental_LR_Scheduler


# Global definitions - data
DATA_FN = 'data/crowdflower_data.csv'
LABEL_NAMES = ["happiness", "worry", "neutral", "sadness"]

# Global definitions - architecture
EMBEDDING_DIM = 100  # We will use pretrained 100-dimensional GloVe
BATCH_SIZE = 128
NUM_CLASSES = 4
USE_CUDA = torch.cuda.is_available()  # CUDA will be available if you are using the GPU image for this homework

# Global definitions - saving and loading data
FRESH_START = False  # set this to false after running once with True to just load your preprocessed data from file
#                     (good for debugging)
TEMP_FILE = "temporary_data.pkl"  # if you set FRESH_START to false, the program will look here for your data, etc.

def train_model(model, loss_fn, optimizer,
                train_generator, dev_generator,
                lr_scheduler=None):
    """
    Perform the actual training of the model based on the train and dev sets.
    :param model: one of models, to be trained to perform emotion classification
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param optimizer: an optimizer you will use to update your model weights
    :param train_generator: a DataLoader that provides batches of the training set
    :param dev_generator: a DataLoader that provides batches of the development set
    :return model, the trained model
    """

    consecutive_negative = 0  # Variable to check patience for early stopping
    development_loss = 100.0
    # Train for each epoch
    for n in range(20):
        """
        Loop through the train dataset to perfom
        bacth optimization
        """
        for x, y in train_generator:
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        # Check the loss of trained model on dev dataset
        loss = test_model(model, loss_fn, dev_generator)

        # Print Loss on Dev set for each iteration
        print('Epoch', n + 1, 'Loss:', loss.item())

        # Return best model if loss doesnt increase consecutively
        # Implementation of early stopping
        if development_loss < loss and consecutive_negative > 2:
            return best_model
        # Maintain
        elif development_loss < loss:
            consecutive_negative += 1
        else:
            development_loss = loss
            consecutive_negative = 0
            # Copy model in another variable if loss keeps on decreasing
            best_model = copy.deepcopy(model)

        # Call learning rate scheduler to change learning rate based on loss
        if lr_scheduler is not None:
            lr_scheduler.update_learing_rate(optimizer, n, loss, development_loss)

    return best_model


def test_model(model, loss_fn, test_generator):
    """
    Evaluate the performance of a model on the development set, providing the loss and macro F1 score.
    :param model: trained model
    :param loss_fn: a function that can calculate loss between the predicted and gold labels
    :param test_generator: a DataLoader that provides batches of the testing set
    """
    gold = []
    predicted = []

    # Keep track of the loss
    loss = torch.zeros(1)  # requires_grad = False by default; float32 by default
    if USE_CUDA:
        loss = loss.cuda()

    model.eval()

    # Iterate over batches in the test dataset
    with torch.no_grad():
        for X_b, y_b in test_generator:
            # Predict
            y_pred = model(X_b)

            # Save gold and predicted labels for F1 score - take the argmax to convert to class labels
            gold.extend(y_b.cpu().detach().numpy())
            predicted.extend(y_pred.argmax(1).cpu().detach().numpy())

            loss += loss_fn(y_pred.double(), y_b.long()).data

    # Print total loss and macro F1 score
    print("Test loss: ")
    print(loss)
    print("F-score: ")
    print(f1_score(gold, predicted, average='macro'))

    return loss

def main(args):
    """
    Train and test neural network models for emotion classification.
    """
    # Prepare the data and the pretrained embedding matrix
    if FRESH_START:
        print("Preprocessing all data from scratch....")
        train, dev, test = utils.get_data(DATA_FN)
        # train_data includes .word2idx and .label_enc as fields
        train_generator, dev_generator, test_generator, embeddings, train_data = utils.vectorize_data(train, dev, test,
                                                                                                      BATCH_SIZE,
                                                                                                      EMBEDDING_DIM)

        print("Saving DataLoaders and embeddings so you don't need to create them again; you can set FRESH_START to "
              "False to load them from file....")
        with open(TEMP_FILE, "wb+") as f:
            pickle.dump((train_generator, dev_generator, test_generator, embeddings, train_data), f)
    else:
        try:
            with open(TEMP_FILE, "rb") as f:
                print("Loading DataLoaders and embeddings from file....")
                train_generator, dev_generator, test_generator, embeddings, train_data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("You need to have saved your data with FRESH_START=True once in order to load it!")

    # Use this loss function in your train_model() and test_model()
    loss_fn = nn.CrossEntropyLoss()

    torch.manual_seed(0)
    model = args.model

    # Train a Dense Network and check F1 score
    if model == 'dense':
        dense_network_model = models.DenseNetwork(100, 10, 4, embeddings)
        optimizer = optim.Adam(dense_network_model.parameters(), lr=0.001)
        train_model(dense_network_model, loss_fn, optimizer, train_generator, dev_generator)
        test_model(dense_network_model, loss_fn, test_generator)

    # Train a RNN(GRU) and check F1 score
    elif model == 'RNN':
        rnn_model = models.RecurrentNetwork(100, 100, 2, 4, embeddings)
        optimizer = optim.Adam(rnn_model.parameters(), lr=0.007, weight_decay=0.00001)
        train_model(rnn_model, loss_fn, optimizer, train_generator, dev_generator)
        test_model(rnn_model, loss_fn, test_generator)

    # Train a CNN and check F1 score.
    # Train CNN with Learning Rate Scheduler Enabled
    elif model == 'CNN':
        cnn_model = models.CNN(embeddings, 100, [2], 4)
        optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
        lr_scheduler = Experimental_LR_Scheduler(2)
        train_model(cnn_model, loss_fn, optimizer,
                    train_generator, dev_generator,
                    lr_scheduler)
        test_model(cnn_model, loss_fn, test_generator)

    # Train a Dense Network with Learning Rate Scheduler Enabled
    # and check F1 score.
    elif model == 'extension':
        dense_network_model = models.DenseNetwork(100, 10, 4, embeddings)
        optimizer = optim.Adam(dense_network_model.parameters(), lr=0.001)
        lr_scheduler = Experimental_LR_Scheduler(2)
        train_model(dense_network_model, loss_fn, optimizer,
                    train_generator, dev_generator,
                    lr_scheduler)
        test_model(dense_network_model, loss_fn, test_generator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', required=True,
                        choices=["dense", "RNN", "CNN", "extension"],
                        help='The name of the model to train and evaluate.')
    args = parser.parse_args()
    main(args)
