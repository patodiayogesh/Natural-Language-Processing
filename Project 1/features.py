import re
from collections import Counter
from copy import deepcopy

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder


# Function to Separate Data based on Religion & Other
def separate_categories(data):
    religion_data = data[data.category == 'Religion']
    remaining_data = data[data.category != 'Religion']
    return religion_data, remaining_data


# Function to remove whitespaces
def clean_text(text):
    # Convert to lower case
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    return text


# Calculate Average VAD Score for a debater
def calculate_average_lexical_scores(text, lexicon_data):
    # Get count of each text
    text_count = Counter(text.split())
    # Store word in Dataframe along with count
    text_df = pd.DataFrame(list(text_count.items()),
                           columns=["Word", "frequency"])
    # Merge lexicon data and Word frequency data to get corresponding scores for debater
    df2 = pd.merge(text_df, lexicon_data, on=['Word', 'Word'])
    # Calculate Average scores individually
    df2['vsum'] = df2['frequency'] * df2['Valence']
    df2['asum'] = df2['frequency'] * df2['Arousal']
    df2['dsum'] = df2['frequency'] * df2['Dominance']
    frequency_sum = df2['frequency'].sum()
    # Case to handle when data in a side is missing
    if frequency_sum == 0:
        return 0, 0, 0
    return (df2['vsum'].sum() / frequency_sum,
            df2['asum'].sum() / frequency_sum,
            df2['dsum'].sum() / frequency_sum)


# Function to Clean and Unpack Data in Each Row
def clean_data(rounds_data, lexicon_data):
    # join text of each round
    clean_values = []
    text = {"Pro": '', "Con": ''}
    for round in rounds_data:
        for side in round:
            try:
                text[side["side"]] += clean_text(side['text']) + ' '
            except:  # Exception Handled when text does not have a suitable key for identification
                pass

    # Append Clean Text Data Separately for Pro and Con
    clean_values.append(text["Pro"])
    clean_values.append(text["Con"])
    clean_values.append(text["Pro"] + ' ' + text["Con"])

    # Calculate Average VAD score each for Pro
    pro_valence_score, pro_arousal_score, pro_dominance_score = \
        calculate_average_lexical_scores(text["Pro"],
                                         lexicon_data)

    clean_values.append(pro_valence_score)
    clean_values.append(pro_arousal_score)
    clean_values.append(pro_dominance_score)

    # Calculate Average VAD score each for Pro
    con_valence_score, con_arousal_score, con_dominance_score = \
        calculate_average_lexical_scores(text["Con"],
                                         lexicon_data)

    clean_values.append(con_valence_score)
    clean_values.append(con_arousal_score)
    clean_values.append(con_dominance_score)

    # Return all new data as series to be appended to each row in the data table
    text = pd.Series(clean_values)
    return text


# Function to Calculate the no. of Reference to websites made by each debater
def get_reference_features(data_train):
    pro_links = pd.DataFrame(data_train['pro_text'].str.count('http|www'))
    con_links = pd.DataFrame(data_train['con_text'].str.count('http|www'))

    return pro_links, con_links


# Function to Calculate the no. of Personal Pronouns Mentioned by each debater
def get_pronouns(data_train):
    # List of personal_pronouns
    personal_pronouns = 'i|you|he|she|it|we|they|me|him|her|us|himself|herself|them'

    # Obtain Count of Pronoun
    pro_pos = pd.DataFrame(data_train['pro_text'].str.count(personal_pronouns))
    con_pos = pd.DataFrame(data_train['con_text'].str.count(personal_pronouns))

    return pro_pos, con_pos


# Function to get Debater Religion Data From User Table
def get_ethnicity_data(encoder, data, user_data):
    # Merge the dataframes to get the data pertaining to each debater from user table
    # Extracting Religious Ideology Data for each debater
    pro_data = pd.merge(data, user_data, how='inner', on='pro_debater')[['ethnicity']]
    con_data = pd.merge(data, user_data, how='inner', on='con_debater')[['ethnicity']]

    # Encode the data
    pro_ethnicity = encoder.transform(pro_data)
    con_ethnicity = encoder.transform(con_data)

    return pro_ethnicity, con_ethnicity


# Function to get Debater Education Data From User Table
def get_political_data(encoder,data_train,user_data):
    # Extracting Political Ideology Data for each debater
    pro_data = pd.merge(data_train, user_data, how='inner', on='pro_debater')[['political_ideology']]
    con_data = pd.merge(data_train, user_data, how='inner', on='con_debater')[['political_ideology']]

    # Encode the data
    pro_political = encoder.transform(pro_data)
    con_political = encoder.transform(con_data)

    return pro_political, con_political


# Function to Get TF-IDF Vector of N-Grams
def get_ngrams(data, test_data, X_train, X_test):
    # TFIDF Vectorizer
    tfid_vectorizer = TfidfVectorizer(lowercase=True,
                                      analyzer='word',
                                      ngram_range=(1, 1)
                                      )
    tfid_vectorizer.fit(data["clean_text"])

    # Get Feature Vector for Train Data
    # Get Feature Vector For Pro Text
    X_train_ngrams = tfid_vectorizer.transform(data['pro_text'])
    X_train = hstack([X_train, X_train_ngrams])

    # Get Feature Vector For Con Text
    X_train_ngrams = tfid_vectorizer.transform(data['con_text'])
    X_train = hstack([X_train, X_train_ngrams])

    # Get Feature Vector for Test Data
    # Get Feature Vector For Pro Text
    X_test_ngrams = tfid_vectorizer.transform(test_data['pro_text'])
    X_test = hstack([X_test, X_test_ngrams])

    # Get Feature Vector For Con Text
    X_test_ngrams = tfid_vectorizer.transform(test_data['con_text'])
    X_test = hstack([X_test, X_test_ngrams])

    # Save Ngram Feature
    return X_train, X_test


# Function to Get Lexicon Feature Vector
def get_lexicon_features(data, X_train):
    # Convert lexicon scores in data table to array
    X_train_lexicons = data[["pro_valence", "pro_arousal", "pro_dominance",
                             "con_valence", "con_arousal", "con_dominance"]].to_numpy()
    X_train = hstack([X_train, X_train_lexicons])
    return X_train


# Get Linguistic Features
def get_linguistic_features(data, X_train):
    # Getting Feature Vector for Data Pertaining to Reference to Websites
    pro_f1, con_f1 = get_reference_features(data)
    X_train = hstack([X_train, pro_f1])
    X_train = hstack([X_train, con_f1])

    # Getting Feature Vector for Data Pertaining to Mention of Personal Pronouns
    pro_f2, con_f2 = get_pronouns(data)
    X_train = hstack([X_train, pro_f2])
    X_train = hstack([X_train, con_f2])

    return X_train


# Get User Features
def get_user_features(data, user_data, X_train):
    # Columns pertaining to user name added as columns for merge action
    user_data['con_debater'] = user_data.index
    user_data['pro_debater'] = user_data.index

    # One Hot Encoder to save Debater Ethnicity Data
    encoder = OneHotEncoder().fit(user_data[['ethnicity']])
    pro_ethnicity, con_ethnicity = get_ethnicity_data(encoder,
                                                   data,
                                                   user_data)
    X_train = hstack([X_train, pro_ethnicity])
    X_train = hstack([X_train, con_ethnicity])

    # One Hot Encoder to save Debater Political Ideology Data
    encoder = OneHotEncoder().fit(user_data[['political_ideology']])
    pro_political, con_political = get_political_data(encoder,
                                                   data,
                                                   user_data)
    X_train = hstack([X_train, pro_political])
    X_train = hstack([X_train, con_political])

    return X_train


# Function to Get Input Feature for Each Model
def process(data, test_data, user_data):
    # Creating an empty array to store feature vectors
    X_train = np.empty((data.shape[0], 0))
    X_test = np.empty((test_data.shape[0], 0))
    feature_label = {}

    # Getting N-Gram Features
    X_train, X_test = get_ngrams(data, test_data, X_train, X_test)
    feature_label['Ngram'] = {'Train':deepcopy(X_train),
                              'Test':deepcopy(X_test)}

    # Getting Lexicon Features
    X_train = get_lexicon_features(data, X_train)
    X_test = get_lexicon_features(test_data, X_test)
    feature_label['Lex'] = {'Train':deepcopy(X_train),
                            'Test':deepcopy(X_test)}

    # Getting Linguistic Features
    X_train = get_linguistic_features(data, X_train)
    X_test = get_linguistic_features(test_data, X_test)
    feature_label['Ling'] = {'Train':deepcopy(X_train),
                            'Test':deepcopy(X_test)}

    # Getting User Features
    X_train = get_user_features(data, user_data, X_train)
    X_test = get_user_features(test_data, user_data, X_test)
    feature_label['User'] = {'Train':deepcopy(X_train),
                            'Test':deepcopy(X_test)}

    return feature_label

#Function to read feature vector for fast loading on consecutive runs
def read_feature_file():
    try:
        with open('yp2607_feature.pickle', 'rb') as feature_file:
            feature = pickle.load(feature_file)
    except FileNotFoundError:
        return None, False
    except Exception:
        return None, False

    if 'User' in feature:
        return feature, True



# Function to Read Files and return Input Features for Model
def get_features(train_path, test_path, lexicon_path, user_path):
    # Read Files
    data = pd.read_json(train_path, lines=True)
    user_data = pd.read_json(user_path).T
    lexicon_data = pd.read_csv(lexicon_path + 'NRC-VAD-Lexicon-Aug2018Release/NRC-VAD-Lexicon.txt',
                               sep='\t', header=None,
                               )

    y_train = None
    if 'winner' in data.columns:
        y_train = data['winner']

    feature_labels,check = read_feature_file()
    if check:
        return feature_labels, y_train

    lexicon_data.columns = ['Word', 'Valence', 'Arousal', 'Dominance']

    test_data = pd.read_json(test_path, lines=True)

    # Clean and Unpack Data
    new_columns = ["pro_text", "con_text", "clean_text",
                   "pro_valence", "pro_arousal", "pro_dominance",
                   "con_valence", "con_arousal", "con_dominance", ]

    data[new_columns] = data["rounds"].apply(lambda x:
                                             clean_data(x, lexicon_data)
                                             )
    test_data[new_columns] = test_data["rounds"].apply(lambda x:
                                                clean_data(x, lexicon_data)
                                                )

    # Get Input Features From Data
    data_train = data.loc[:, data.columns != 'winner']

    feature_labels = process(data_train, test_data, user_data)

    with open('yp2607_feature.pickle','wb') as feature_file:
        pickle.dump(feature_labels, feature_file, protocol=pickle.HIGHEST_PROTOCOL)

    return feature_labels, y_train