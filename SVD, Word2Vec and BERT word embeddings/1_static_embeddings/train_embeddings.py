'''
Embedding Creation
'''

import re
import timeit
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from gensim.models import Word2Vec

CORPUS_TXT = './data/brown.txt'


class SVD_PPMI:
    """
    Class to generate PPMI SVD based embeddings

    Attributes:
    -----------------
    text : Uncleaned text
    unigrams: dictionary containing unigrams and their count
    word_idx: dictionary containing unigrams and tbeir index
    cleaned_text: cleaned corpus of data
    """
    def __init__(self):
        """
        Function to initialize the class variables
        """
        self.text = []
        self.unigrams = None
        self.word_idx = None
        self.cleaned_text = []


    def read_file(self):
        """
        Function to read the file from the disk
        Return a list of sentences
        """
        if not self.text == []:
            return self.text

        with open(CORPUS_TXT, 'r') as f:
            sentences = f.read().splitlines()
        return sentences

    def get_skipgrams(self,sentences, window=2):
        """
        Function to pairs of words in the context window window

        :param sentences: cleaned text
        :param window: context window size
        :return: dictionary of words with their counts
        """
        skip_grams = defaultdict(lambda: 0)
        for sentence in sentences:
            sentence_length = len(sentence)
            # Get pair of words for a given word in the sentence
            # Increase counter for each pair of word
            for index, word in enumerate(sentence):
                for w in range(1, window + 1):
                    if index + w < sentence_length:
                        skip_grams[(word, sentence[index + w])] += 1
                    if index - w >= 0:
                        skip_grams[(word, sentence[index - w])] += 1
        return skip_grams

    def get_unigrams(self,sentences):
        """
        Function cleans the text and saves it for future use
        Function to get the unigrams and
        their counts and index in co-occurence matrix.

        :param sentences: Uncleaned text
        :return unigrams: dictionary of unigrams with their counts
        :return word_idx: dictionary of unigrams with their index
        """

        # If cleaned text is present return unigrams and word_idx
        if not self.text == []:
            return self.unigrams, self.word_idx

        unigrams = defaultdict(lambda: 0)
        word_idx = {}
        count = 0
        for index, sentence in enumerate(sentences):

            # Clean text
            sentence = re.sub(r'[^\w\s+]', '', sentence.lower().strip())
            sentence = re.sub(' +', ' ', sentence)
            self.cleaned_text.append(sentence)
            sentence = sentence.split(' ')
            self.text.append(sentence)
            sentences[index] = sentence

            # Get unigram count and index
            for word in sentence:
                if not unigrams[word]:
                    word_idx[word] = count
                    count += 1
                unigrams[word] += 1


        self.unigrams = unigrams
        self.word_idx = word_idx

        return self.unigrams, self.word_idx


    def create_pmi_matrix(self,unigrams, skip_grams, word_idx):
        """
        Function to create pmi matrix from
        unigrams, skip_grams(pair of words), word_idx

        :param unigrams: dictionary of unigrams with their counts
        :param word_idx: dictionary of unigrams with their index
        :param skip_grams: dictionary of skip_grams with their counts

        :return sparse_matrix: Sparse matrix which is the PPMI matrix
        """

        unigram_sum = sum(unigrams.values())
        skip_gram_sum = sum(skip_grams.values())
        scale = skip_gram_sum
        # Create empty sparse matrix
        sparse_matrix = sparse.lil_matrix((len(unigrams), len(unigrams)))

        """
        Utilize unigram word_idx to create ppmi matrix for
        pair of words that were encountered.
        """
        for bigram, count in skip_grams.items():
            word_x = bigram[0]
            word_y = bigram[1]
            sparse_matrix[word_idx[word_x], word_idx[word_y]] = \
                np.log(count / (unigrams[word_x] * unigrams[word_y]) * scale)
        return sparse_matrix

    def create_save_embeddings(self,model_number,
                               n_components=128,
                               context_window=2):
        """
        Function to apply SVD on PPMI matrix
        to get word embeddings and save them in file
        Calls functions to generte ppmi matrix

        :param model_number: Number to identify embedding variations
        :param n_components: Dimension
        :param context_window: Context window

        :return None
        """

        sentences = self.read_file()
        unigrams, word_idx = self.get_unigrams(sentences)
        skipgrams = self.get_skipgrams(sentences, window=context_window)
        sparse_matrix = self.create_pmi_matrix(unigrams, skipgrams, word_idx)

        # Pass sparse matrix to TruncatedSVD
        svd_matrix = TruncatedSVD(n_components=n_components, random_state=42)
        svd_matrix.fit(sparse_matrix)
        # Use transform to generate word embeddings
        word_embeddings = svd_matrix.transform(sparse_matrix)
        # Save embeddings in file
        svd_embeddings = 'svd_embeddings_'+str(model_number)+'.txt'
        with open(svd_embeddings, 'w') as f:
            for word in unigrams:
                f.write(word + ' ' + str(word_idx[word]) + ' ' + \
                        ' '.join([str(x) for x in word_embeddings[word_idx[word]]]) + \
                        '\n')

class Word2Vec_Skipgram:
    """
    Class to generate Word2Vec based embeddings

    Attributes:
    -----------------
    text : Uncleaned text
    model: Word2Vec model with different parameters
    """

    def __init__(self):
        """
        Function to save the cleaned text and model
        """
        self.text = None
        self.model = None

    def create_embeddings(self,model_number):
        """
        Function to utilise Word2Vec model with
        different parameters to generate word embeddings
        and save them

        :param model_number: Number to identify embedding variations
        """
        # Utilising Word2Vec functions to generate and save word embeddings
        self.model.build_vocab(self.text)
        self.model.train(self.text,
                         total_examples=self.model.corpus_count,
                         epochs=5)
        self.model.save('word2vec_'+str(model_number)+'.wv')

if __name__ == '__main__':

    # Create SVD_PPMI object
    embeddings = SVD_PPMI()

    # Variation 1
    print('Creating Model 1')
    print(timeit.timeit())
    embeddings.create_save_embeddings(1,100,2)
    print(timeit.timeit())
    # Variation 2
    print('Creating Model 2')
    print(timeit.timeit())
    embeddings.create_save_embeddings(2,300,2)
    print(timeit.timeit())
    # Variation 3
    print('Creating Model 3')
    print(timeit.timeit())
    embeddings.create_save_embeddings(5, 50, 5)
    print(timeit.timeit())
    # Variation 4
    print('Creating Model 4')
    print(timeit.timeit())
    embeddings.create_save_embeddings(3, 100, 10)
    print(timeit.timeit())
    # Variation 5
    print('Creating Model 5')
    print(timeit.timeit())
    embeddings.create_save_embeddings(4, 300, 10)
    print(timeit.timeit())

    # Create Word2Vec_Skipgram object
    word2vec_model = Word2Vec_Skipgram()
    word2vec_model.text = embeddings.text
    # 5 Variations of parameters for word2vec model
    params = [{'vector_size':100,'window':2,'min_count':1,'workers':4,'negative':1,'sg':1},
              {'vector_size':300,'window':2,'min_count':1,'workers':4,'negative':1,'sg':1},
              {'vector_size': 50, 'window': 5, 'min_count': 1, 'workers': 4, 'negative': 1, 'sg': 15},
              {'vector_size':100,'window':10,'min_count':1,'workers':4,'negative':1,'sg':1},
              {'vector_size':300,'window':10,'min_count':1,'workers':4,'negative':1,'sg':15},
              ]
    for index,param in enumerate(params):
        word2vec_model.model = Word2Vec(**param)
        word2vec_model.create_embeddings(index+1)