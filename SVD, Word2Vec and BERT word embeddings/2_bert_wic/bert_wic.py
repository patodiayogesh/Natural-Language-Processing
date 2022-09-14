'''
TO create bert word embeddings and get word context similarity
'''

import os
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from transformers import BertModel, BertTokenizer

from sklearn.metrics import accuracy_score, classification_report

from models import DenseNetwork, RecurrentNetwork, CNN
from models import train_model
from models import ExperimentalLRScheduler

LABELS = ['F', 'T']


def get_wic_subset(data_dir):
    wic = []
    split = data_dir.strip().split('/')[-1]
    with open(os.path.join(data_dir, '%s.data.txt' % split), 'r', encoding='utf-8') as datafile, \
            open(os.path.join(data_dir, '%s.gold.txt' % split), 'r', encoding='utf-8') as labelfile:
        for (data_line, label_line) in zip(datafile.readlines(), labelfile.readlines()):
            word, _, word_indices, sentence1, sentence2 = data_line.strip().split('\t')
            sentence1_word_index, sentence2_word_index = word_indices.split('-')
            label = LABELS.index(label_line.strip())
            wic.append({
                'word': word,
                'sentence1_word_index': int(sentence1_word_index),
                'sentence2_word_index': int(sentence2_word_index),
                'sentence1_words': sentence1.split(' '),
                'sentence2_words': sentence2.split(' '),
                'label': label
            })
    return wic


def save_predicted_data(predicted_labels, out_file):
    """
    Function to write the predicted data onto a file.

    :param predicted_labels: Predicted output
    :param out_file: Path where output should be written
    """
    with open(out_file, mode='w', encoding='utf-8') as output_file:
        for p in predicted_labels:
            if p:
                output_file.write('T\n')
            else:
                output_file.write('F\n')
    return True


class WordSimilarity:
    """
    Class to generate tokens and word embeddings
    using BERT transformers and tokenizers
    The word embeddings are passed through a Dense Neural Network
    The DNN is used to predict the similarity in word context
    """
    def __init__(self, classifier=None, ):
        """
        Function to initialize the Bert tokenizer and model
        """
        self.classifier = classifier

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                       do_lower_case=True)
        self.model = BertModel.from_pretrained('bert-base-cased')
        self.model.eval()

    def get_tokens_and_label(self, dataset):
        """
        Function to generate sentence tokens and word embeddings
        It also gets the corresponding gold label from the dataset

        :param dataset: Dataset
        """

        def get_target_word_index(word_index, words):
            """
            Function to get the target word and sub-words index
            :param word_index: word index in sentence
            :param words: list of words that are tokenized

            :return start_index: Start index of target token
            :return target_word_length: No of subwords
            """
            start_index = 1
            # Get start_index
            for x in words[:word_index]:
                temp_tokens = self.tokenizer.tokenize(x)
                # tokens.append(temp_tokens)
                start_index += len(temp_tokens)
            # Get number of word and sub-words
            temp_tokens = self.tokenizer.tokenize(words[word_index])
            target_word_length = len(temp_tokens)

            return start_index, target_word_length


        embeddings = []
        labels = []

        with torch.no_grad():
            for sentence in dataset:
                s1_index = sentence['sentence1_word_index']
                s2_index = sentence['sentence2_word_index']

                s1 = ' '.join(sentence['sentence1_words'])
                s2 = ' '.join(sentence['sentence2_words'])

                # Get word tokens for a pair of sentence
                tokenized = self.tokenizer([s1, s2], padding=True,
                                           return_tensors='pt')

                # Get word embeddings for the generated tokens
                output = self.model(**tokenized)

                # Get target word indices
                word1_index, w1_len = get_target_word_index(s1_index,
                                                            sentence['sentence1_words']
                                                            )
                word2_index, w2_len = get_target_word_index(s2_index,
                                                            sentence['sentence2_words']
                                                            )

                # Get word embeddings of target words by taking meanher
                # Stach the embeddings toget
                embedding = torch.hstack([torch.mean(output.last_hidden_state[0][word1_index:word1_index + w1_len],
                                                     dim=0),
                                          torch.mean(output.last_hidden_state[1][word2_index:word2_index + w2_len],
                                                     dim=0)]
                                         ).numpy()
                embeddings.append(embedding)

                # Gold Label
                labels.append(sentence['label'])

        return np.array(embeddings), np.array(labels)

    def set_classifier(self, model, params):
        """
        Function to set classifier for prediction

        :param model: String indicating model name
        :param params: Paramets for corresponding model
        """

        if model == 'DenseNetwork':
            self.classifier = DenseNetwork(**params)
            return

        if model == 'CNN':
            self.classifier = CNN(**params)
            return

    def fit(self, dataset=None, labels=None):

        """
        Function to fit the model to predict similarity in word context
        Calls function to generate embeddings and labels
        """
        if labels is None or dataset is None:
            embeddings, labels = self.get_tokens_and_label(dataset)
        else:
            embeddings = dataset

        embeddings = torch.Tensor(embeddings)
        labels = torch.Tensor(labels).type(torch.LongTensor)

        # Create Tensor Train Dataset
        train_dataset = TensorDataset(embeddings[:5000], labels[:5000])
        train_loader = DataLoader(train_dataset, batch_size=128)
        # Create Tensor Dev Dataset
        dev_dataset = TensorDataset(embeddings[5000:5428], labels[5000:5428])
        dev_loader = DataLoader(dev_dataset, batch_size=128)

        # Train model
        loss_fn = nn.CrossEntropyLoss()
        lr_scheduler = ExperimentalLRScheduler(0)
        # 128, .002
        # dense_network_model = RecurrentNetwork(1536, 128, 4, 2)
        # dense_network_model = CNN(100, [2], 2)
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.002)
        train_model(self.classifier, loss_fn,
                    optimizer, train_loader,
                    dev_loader,
                    lr_scheduler,
                    )

    def predict(self, dataset):
        """
        Function to predict the data using trained model

        :param dataset: Dev Dataset

        :return: predicted labels
        """
        dataset = torch.Tensor(dataset)
        test_loader = DataLoader(dataset, batch_size=128)
        
        predicted_labels = []
        with torch.no_grad():
            for X_b in test_loader:
                y_pred = self.classifier(X_b)
                predicted_labels.extend(y_pred.argmax(1).cpu().detach().numpy())
        return predicted_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a classifier to recognize words in context (WiC).'
    )
    parser.add_argument(
        '--train-dir',
        dest='train_dir',
        required=True,
        help='The absolute path to the directory containing the WiC train files.'
    )
    parser.add_argument(
        '--eval-dir',
        dest='eval_dir',
        required=True,
        help='The absolute path to the directory containing the WiC eval files.'
    )

    parser.add_argument(
        '--out-file',
        dest='out_file',
        required=True,
        help='The absolute path to the file where evaluation predictions will be written.'
    )
    args = parser.parse_args()

    torch.manual_seed(0)

    # Load train and test dataset from disk
    train_dataset = get_wic_subset(args.train_dir)
    test_dataset = get_wic_subset(args.eval_dir)

    # Generate Word Embeddings and the gold labels for each
    wic_similarity = WordSimilarity()
    train_x, train_y = wic_similarity.get_tokens_and_label(train_dataset)
    test_x, test_y = wic_similarity.get_tokens_and_label(test_dataset)

    # Parameters of Dense Neural Network which is set for training
    dense_network_params = {'input_size': 1536,
                            'hidden_size': 128,
                            'num_classes': 2
                            }
    wic_similarity.set_classifier('DenseNetwork',
                                  dense_network_params,
                                  )
    wic_similarity.fit(train_x, train_y)

    predicted = wic_similarity.predict(test_x)
    # Write predicted data to out file
    save_predicted_data(predicted, args.out_file)

    print(classification_report(test_y, predicted))
    print(accuracy_score(test_y, predicted))
