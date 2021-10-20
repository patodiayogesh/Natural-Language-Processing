"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File

<YOUR NAME HERE>
<YOUR UNI HERE>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn



class DenseNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, pre_trained_embeddings):
        super(DenseNetwork, self).__init__()
        self.embedding_layer = nn.Embedding(pre_trained_embeddings.shape[0],
                                            pre_trained_embeddings.shape[1])
        self.embedding_layer.weight.data.copy_(pre_trained_embeddings)
        self.pooling_layer = nn.LPPool2d(norm_type=1,kernel_size=(91,1), stride = 1)
        self.layer1 = nn.Linear(input_size,hidden_size)
        self.activation_func = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class
        output = self.embedding_layer(x)
        pooling_layer = nn.LPPool2d(norm_type=1, kernel_size=(x.size(1), 1), stride=1)
        output = pooling_layer(output)
        output = output.squeeze()
        output = self.layer1(output)
        output = self.activation_func(output)
        output = self.layer2(output)
        return output.squeeze()


class RecurrentNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, pre_trained_embeddings):
        super(RecurrentNetwork, self).__init__()

        self.embedding_layer = nn.Embedding(pre_trained_embeddings.shape[0],
                                            pre_trained_embeddings.shape[1]).\
                                from_pretrained(embeddings=pre_trained_embeddings)
        # self.embedding_layer.weight.data.copy_(pre_trained_embeddings)
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=layer_dim, batch_first=True)
        self.func = nn.Linear(hidden_dim, output_dim)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class

        output = self.embedding_layer(x).double()
        #h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        #c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        output, h1 = self.gru(output)#, h0)
        output = self.func(output[:, -1, :])
        return output


# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    def __init__(self,
                 pre_trained_embeddings,
                 num_filters,
                 filter_sizes,
                 output_dim,
                 dropout):
        super(ExperimentalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embedding_layer = nn.Embedding(pre_trained_embeddings.shape[0], pre_trained_embeddings.shape[1])
        self.embedding_layer.weight.data.copy_(pre_trained_embeddings)
        self.conv_layer = nn.ModuleList([nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(fs, 100)) for fs in filter_sizes])
        self.layer1 = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout_layer = nn.Dropout(dropout)

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        output = self.embedding_layer(x)
        output = output.unsqueeze(1)
        output = [F.relu(conv(output)).squeeze(3) for conv in self.conv_layer]
        output = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in output]
        output = self.dropout_layer(torch.cat(output, dim=1))
        output = self.layer1(output)
        return output
