'''
Exploring BERT each hidden layer relation with syntactic
and semantic sentence information
'''

import torch
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def get_repr(dataset):
    """
    Function to get hidden layer representation for conll
    """
    repr = [[] for _ in range(13)]
    for sentence in dataset:

        hidden_states = sentence['hidden_states']
        word_token_indices = sentence['word_token_indices']
        # Get word representations for each layer
        for layer in range(0, 13):
            hidden_state = hidden_states[layer]
            for i, index in enumerate(word_token_indices):
                index_start, index_last = index[0], index[-1]
                word_rep = torch.mean(hidden_state[index_start:index_last + 1],
                                      dim=0).numpy()

                repr[layer].append(word_rep)

    return repr

def get_sem_repr_label(dataset):
    """
    Function to get hidden layer representation for semeval
    Gets the entities_representation and their label from dataset

    :return repr: Corresponding Embeddings of each layer
    :return rel_label: Gold Label
    """
    repr = [[] for _ in range(13)]
    rel_label = []
    for sentence in dataset:

        entity_1_hidden_states = sentence['entity1_representations']
        entity_2_hidden_states = sentence['entity2_representations']
        rel_label.append(sentence['rel_label'])
        # Get each entity representation and stack them.
        # Operation done for each layer
        for layer in range(0, 13):

            entity_1_hidden_state = entity_1_hidden_states[layer]
            entity_2_hidden_state = entity_2_hidden_states[layer]
            repr[layer].append(torch.hstack([entity_1_hidden_state,
                                            entity_2_hidden_state]).numpy()
                               )
    return repr, np.array(rel_label)

def get_rel_label(dataset):
    rel_label = []
    for sentence in dataset:
        rel_label.append(sentence['rel_label'])

    return np.array(rel_label)

def get_pos_ner_label(dataset):
    """
    Function to get the corresponding gold pos and ner label
    for each word in sentence

    :return pos_label: GOLD POS label
    :return ner_label: GOLD NER label
    """
    pos_label, ner_label = [], []
    for sentence in dataset:
        ner_labels = sentence['ner_labels']
        pos_labels = sentence['pos_labels']
        word_token_indices = sentence['word_token_indices']

        for i, _ in enumerate(word_token_indices):
            pos_label.append(pos_labels[i])
            ner_label.append(ner_labels[i])

    return np.array(pos_label), np.array(ner_label)


def analyze(datset):
    """
    Function to get F1 scores for conll dataset
    Gets the corresponding gold labels and word representations
    for each layer and word

    :return pos_f1_scores: list of POS F1 scores for each layer
    :return ner_f1_scores: list of NER F1 scores for each layer
    """

    # Get labels and representations
    train_pos_label, train_ner_label = get_pos_ner_label(datset['train'])
    test_pos_label, test_ner_label = get_pos_ner_label(datset['validation'])
    train_x = get_repr(datset['train'])
    test_x = get_repr(datset['validation'])

    pos_f1_scores = []
    ner_f1_scores = []

    # For each layer, train and test model to get f1 score
    for layer in range(13):
        clf_pos = LogisticRegression(random_state=0, max_iter=1000). \
            fit(train_x[layer], train_pos_label)
        clf_ner = LogisticRegression(random_state=0, max_iter=1000). \
            fit(train_x[layer], train_ner_label)

        pos_f1_scores.append(f1_score(test_pos_label,
                                      clf_pos.predict(test_x[layer]),
                                      average='macro'))
        ner_f1_scores.append(f1_score(test_ner_label,
                                      clf_ner.predict(test_x[layer]),
                                      average='macro'))

    return pos_f1_scores, ner_f1_scores

def sem_analyze(dataset):
    """
    Function to get F1 scores for semeval dataset
    Gets the corresponding gold labels and word representations
    for each layer and entity

    :return rel_f1_scores: list of REL F1 scores for each layer
    """

    # Get labels and representations
    train_x, train_label = get_sem_repr_label(dataset['train'])
    test_x, test_label = get_sem_repr_label(dataset['test'])

    rel_f1_scores = []
    # For each layer, train and test model to get f1 score
    for layer in range(13):
        clf_rel = LogisticRegression(random_state=0, max_iter=2000). \
            fit(train_x[layer], train_label)

        rel_f1_scores.append(f1_score(test_label,
                                      clf_rel.predict(test_x[layer]),
                                      average='macro'))

    return rel_f1_scores

def plot_f1_scores(pos_f1_scores, ner_f1_scores,rel_f1_scores):
    """
    Function to plot all the F1 scores against the layer
    """
    plt.plot(range(1, 14), pos_f1_scores, color='orange', label='Macro f1 (POS)')
    plt.plot(range(1, 14), ner_f1_scores, color='red', label='Macro f1 (NER)')
    plt.plot(range(1, 14), rel_f1_scores, color='blue', label='Macro f1 (REL)')
    plt.xlabel('Layer')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    plt.title('f1 scores')
    plt.savefig('f1_scores.png')
    plt.show()


if __name__ == '__main__':

    # Load Dataset
    conll_dataset = torch.load('./data/conll.pt')
    sem_dataset = torch.load('./data/semeval.pt')
    # Calculate F1 scores
    pos_f1_scores, ner_f1_scores = analyze(conll_dataset)
    rel_f1_scores = sem_analyze(sem_dataset)
    # Plot
    plot_f1_scores(pos_f1_scores, ner_f1_scores,rel_f1_scores)

