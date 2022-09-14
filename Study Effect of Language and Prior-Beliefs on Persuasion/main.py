import argparse

import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning


from features import get_features

def predict_features(args):
    feature_labels, y_train = get_features(args.train, args.test, args.lexicon_path, args.user_data)
    models = args.model
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    predicted = []
    if models == 'Ngram':
        clf = LogisticRegression()
        clf.fit(feature_labels['Ngram']['Train'], y_train)
        predicted = clf.predict(feature_labels['Ngram']['Test'])

    elif models == 'Ngram+Lex':
        clf = LogisticRegression()
        clf.fit(feature_labels['Lex']['Train'], y_train)
        predicted = clf.predict(feature_labels['Lex']['Test'])

    elif models == 'Ngram+Lex+Ling':
        clf = LogisticRegression(max_iter=300)
        clf.fit(feature_labels['Ling']['Train'], y_train)
        predicted = clf.predict(feature_labels['Ling']['Test'])

    elif models == 'Ngram+Lex+Ling+User':
        clf = LogisticRegression(max_iter=100)
        clf.fit(feature_labels['User']['Train'], y_train)
        predicted = clf.predict(feature_labels['User']['Test'])
    

    predicted = predicted.tolist()
    with open(args.outfile,mode='w', encoding='utf-8') as f:
        f.write('\n'.join(predicted))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', required=True,
                        help='Full path to the training file')
    parser.add_argument('--test', dest='test', required=True,
                        help='Full path to the evaluation file')
    parser.add_argument('--user_data', dest='user_data', required=True,
                        help='Full path to the user data file')
    parser.add_argument('--model', dest='model', required=True,
                        choices=["Ngram", "Ngram+Lex", "Ngram+Lex+Ling", "Ngram+Lex+Ling+User"],
                        help='The name of the model to train and evaluate.')
    parser.add_argument('--lexicon_path', dest='lexicon_path', required=True,
                        help='The full path to the directory containing the lexica.'
                             ' The last folder of this path should be "lexica".')
    parser.add_argument('--outfile', dest='outfile', required=True,
                        help='Full path to the file we will write the model predictions')
    args = parser.parse_args()
    predict_features(args)
