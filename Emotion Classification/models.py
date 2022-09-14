"""
Emotion Classification with Neural Networks - Models File

Yogesh Patodia
yp2607
"""

import torch.nn as nn
import torch.nn.functional as F


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

    def __init__(self, input_size, hidden_size, num_classes, pre_trained_embeddings):
        """
        Initialize the model with arguments
        :param input_size: Input size of Linear layer. Should match the embedding dimension
        :param hidden_size: Hyper parameter
        :param num_classes: Number of labels
        :param pre_trained_embeddings: word embedding vectors
        :return: None
        """

        super(DenseNetwork, self).__init__()
        self.embedding_layer = nn.Embedding(pre_trained_embeddings.shape[0],
                                            pre_trained_embeddings.shape[1])
        # Copy pre trained embeddings into model variable
        self.embedding_layer.weight.data.copy_(pre_trained_embeddings)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation_func = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Function to Train Dense Network Model

        :param x: {tensor} Training data
        :return: {tensor} Output labels

        Trains a 2-layer feedforward network
        which produces a 4-vector of values
        """

        # Get word embedding for corresponding data point
        output = self.embedding_layer(x)
        # Sum Pooling based on input size
        pooling_layer = nn.LPPool2d(norm_type=1, kernel_size=(x.size(1), 1), stride=1)
        output = pooling_layer(output)
        output = output.squeeze()

        # Call Layer1, Activation Function and Layer 2
        output = self.layer1(output)
        output = self.activation_func(output)
        output = self.layer2(output)

        return output.squeeze()


class RecurrentNetwork(nn.Module):
    """
    Implementation of GRU(RNN)

    Attributes:
    -----------
    embedding_layer: layer which saves the word embeddings and
    converts merges train_vector with their embeddings
    hidden_dim: No of nodes in hidden layer
    layer_dim: No. of hidden layers
    gru: GRU Model
    func: Linear layer. Changes dimension to number of labels
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, pre_trained_embeddings):
        """
        Initialize the model with arguments
        :param input_dim: Input size of training data. Should match the embedding dimension
        :param hidden_dim: No of nodes in hidden layer
        :param layer_dim: Number of hidden layers
        :param output_dim: No of Class Labels
        :param pre_trained_embeddings: word embedding vectors
        :return: None
        """
        super(RecurrentNetwork, self).__init__()

        self.embedding_layer = nn.Embedding(pre_trained_embeddings.shape[0],
                                            pre_trained_embeddings.shape[1])
        # Copy pre trained embeddings into model variable
        self.embedding_layer.weight.data.copy_(pre_trained_embeddings)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layer_dim, batch_first=False)
        self.func = nn.Linear(hidden_dim, output_dim, bias=True)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        """
        Function to Train GRU(RNN) Model

        :param x: {tensor} Training data
        :return: {tensor} Output labels
        """

        output = self.embedding_layer(x)
        output, h1 = self.gru(output)
        output = self.func(output[:, -1, :])

        return output


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
                 pre_trained_embeddings,
                 num_filters,
                 filter_sizes,
                 output_dim):
        """
        Initialize the model with arguments

        :param pre_trained_embeddings: word embedding vectors
        :param num_filters: Number of kernels
        :param filter_sizes: Kernel sizes
        :param output_dim: Dimension of Output tensor
        """
        super(CNN, self).__init__()
        self.embedding_layer = nn.Embedding(pre_trained_embeddings.shape[0], pre_trained_embeddings.shape[1])
        self.embedding_layer.weight.data.copy_(pre_trained_embeddings)
        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=num_filters,
                                    kernel_size=(2, 100))
        self.activation_fn = nn.ReLU()
        self.layer1 = nn.Linear(len(filter_sizes) * num_filters, output_dim)

    def forward(self, x):
        """
        Function to Train CNN Network Model

        :param x: {tensor} Training data
        :return: {tensor} Output labels
        """

        output = self.embedding_layer(x)
        output = output.unsqueeze(1)
        output = self.conv_layer(output).squeeze(3)
        # Activation function after convolution
        output = self.activation_fn(output)
        # Pooling the data using max pool
        output = F.max_pool1d(output, output.shape[2]).squeeze(2)
        output = self.layer1(output)
        return output
