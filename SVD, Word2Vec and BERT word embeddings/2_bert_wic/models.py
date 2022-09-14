'''
DNN and CNN model architectures
'''
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
# from sklearn.metrics import f1_score, accuracy_score

USE_CUDA = torch.cuda.is_available()


class ExperimentalLRScheduler:
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
                group['lr'] = group['lr'] * math.exp(-self.gamma *
                                                     epoch_iteration_number)

        return


class DenseNetwork(nn.Module):
    """
    Implementation of Dense Network

    Attributes:
    -----------
    embedding_layer: layer which saves the word embeddings and
    converts merges train_vector with their embeddings
    layer1: Linear layer. Changes dimension based on inputs
    activation_func: ReLU activation function
    layer2: Linear layer. Changes dimension to number of labels
    """

    def __init__(self, input_size, hidden_size, num_classes,
                 # pre_trained_embeddings,
                 ):
        """
        Initialize the model with arguments
        :param input_size: Input size of Linear layer. Should match the embedding dimension
        :param hidden_size: Hyper parameter
        :param num_classes: Number of labels
        :return: None
        """

        super(DenseNetwork, self).__init__()
        # self.embedding_layer = nn.Embedding(pre_trained_embeddings.shape[0],
        #                                     pre_trained_embeddings.shape[1])
        # Copy pre trained embeddings into model variable
        # self.embedding_layer.weight.data.copy_(pre_trained_embeddings)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation_func = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, 64)
        self.layer3 = nn.Linear(64, num_classes)

    def forward(self, x):
        """
        Function to Train Dense Network Model

        :param x: {tensor} Training data
        :return: {tensor} Output labels

        Trains a 2-layer feedforward network
        which produces a 4-vector of values
        """

        # Get word embedding for corresponding data point
        # output = self.embedding_layer(x)
        # Sum Pooling based on input size
        # pooling_layer = nn.LPPool2d(norm_type=1, kernel_size=(x.size(1), 1), stride=1)
        # output = pooling_layer(x)
        # output = output.squeeze()

        # Call Layer1, Activation Function and Layer 2
        output = self.layer1(x)
        output = self.activation_func(output)
        output = self.layer2(output)
        output = self.activation_func(output)
        output = self.layer3(output)

        return output.squeeze()


class CNN(nn.Module):
    """
    Implementation of Experimental Network, CNN

    Attributes:
    -----------
    embedding_layer: layer which saves the word embeddings and
    converts merges train_vector with their embeddings
    conv_layer: Convolution layer. Convoluted data into bigrams
    activation_fn: ReLU Activation Function
    layer1: Linear layer. Changes dimension to number of labels
    """

    def __init__(self,
                 # pre_trained_embeddings,
                 num_filters,
                 filter_sizes,
                 output_dim):
        """
        Initialize the model with arguments

        :param num_filters: Number of kernels
        :param filter_sizes: Kernel sizes
        :param output_dim: Dimension of Output tensor
        """
        super(CNN, self).__init__()

        self.conv_layer = nn.Conv1d(in_channels=1, out_channels=num_filters,
                                    kernel_size=filter_sizes[0])
        self.activation_fn = nn.ReLU()
        self.layer1 = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, x):
        """
        Function to Train CNN Network Model

        :param x: {tensor} Training data
        :return: {tensor} Output labels
        """

        x = x.reshape(x.shape[0], 1, x.shape[1])
        output = self.conv_layer(x)
        # Activation function after convolution
        output = self.activation_fn(output)
        # Pooling the data using max pool
        output = F.max_pool1d(output, output.shape[2]).squeeze(2)
        output = self.layer1(output)
        return output


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
    # print("Test loss: ")
    # print(loss)
    # print("F-score: ")
    # print(f1_score(gold, predicted, average='macro'))
    # print(classification_report(gold, predicted))
    # print(accuracy_score(gold, predicted))

    return loss


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
        if development_loss < loss and consecutive_negative > 3:
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
