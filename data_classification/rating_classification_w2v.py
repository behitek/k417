import os
import pickle
import re
import string
import time

import nltk
import numpy as np
import pandas as pd
import seaborn as sns  # used for plot interactive graph.
from gensim.models import Word2Vec
from imblearn.over_sampling import SMOTE
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

sns.set_style('darkgrid')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.read_csv('../tripadvisor_hotel_reviews.csv')
data = data.astype({"Review": str, "Rating": int})
# stopwords = set(open('../data_understanding/stopwords.txt').read().splitlines())
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
punc = string.punctuation
nltk.download('wordnet')


def preprocess(row):
    # converting to lowercase
    _row = row['Review']
    _row = _row.lower()
    # Removing Punctuation
    _row = re.sub(r'[^a-z ]+', ' ', _row)
    # Removing stopwords
    _row = " ".join([word for word in str(_row).split() if word not in stop_words])
    # Stemming
    _row = " ".join([stemmer.stem(word) for word in _row.split()])
    # Lemmatization
    _row = " ".join([lemmatizer.lemmatize(word) for word in _row.split()])
    return _row.split()


def remake_label(row):
    return row['Rating'] - 1


data['Text'] = data.apply(lambda row: preprocess(row), axis=1)
data['Rating'] = data.apply(lambda row: remake_label(row), axis=1)

X = data['Text']
y = data['Rating']

print('Done pre processing')

w2v_size = 100
word2vec_model_file = 'word2vec_' + str(w2v_size) + '.model'


def train_w2v():
    # Skip-gram model (sg = 1)
    window = 3
    min_count = 1
    workers = 3
    sg = 1
    start_time = time.time()
    stemmed_tokens = pd.Series(data['Text']).values
    # Train the Word2Vec Model
    w2v_model = Word2Vec(stemmed_tokens, min_count=min_count, vector_size=w2v_size, workers=workers, window=window,
                         sg=sg)
    print("Time taken to train word2vec model: " + str(time.time() - start_time))
    w2v_model.save(word2vec_model_file)


if not os.path.exists(word2vec_model_file):
    train_w2v()
sg_w2v_model = Word2Vec.load(word2vec_model_file)

# Total number of the words
print(len(sg_w2v_model.wv.key_to_index))

word2vec_filename = 'train_review_word2vec.csv'

with open(word2vec_filename, 'w+') as word2vec_file:
    for index, row in enumerate(X.tolist()):
        model_vector = (np.mean([sg_w2v_model.wv[token] for token in row], axis=0)).tolist()
        if index == 0:
            header = ",".join(str(ele) for ele in range(w2v_size))
            word2vec_file.write(header)
            word2vec_file.write("\n")
        # Check if the line exists else it is vector of zeros
        if type(model_vector) is list:
            line1 = ",".join([str(vector_element) for vector_element in model_vector])
        else:
            line1 = ",".join([str(0) for i in range(100)])
        word2vec_file.write(line1)
        word2vec_file.write('\n')

X_w2v = pd.read_csv(word2vec_filename)
# SMOTE Technique
oversample = SMOTE()
X, y = oversample.fit_resample(X_w2v, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def train_pipeline(classifier, name='default'):
    text_clf = Pipeline([('clf', classifier)])
    print('Start training {} ...'.format(name))
    text_clf = text_clf.fit(X_train, y_train)
    print('Done training {} ...'.format(name))
    y_pred = text_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Save model
    pickle.dump(text_clf, open("{}.pkl".format(name), "wb"))
    print()


train_pipeline(RandomForestClassifier(max_depth=None), 'random_forest_w2v')
train_pipeline(LogisticRegression(max_iter=10000), 'logistic_regression_w2v')
train_pipeline(XGBClassifier(use_label_encoder=False, base_score=0.5, booster='gbtree', colsample_bylevel=1,
                             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                             importance_type='gain', interaction_constraints='',
                             learning_rate=0.05, max_delta_step=0, max_depth=None,
                             min_child_weight=1, monotone_constraints='()',
                             n_estimators=100, n_jobs=0, num_parallel_tree=1,
                             objective='multi:softprob', random_state=1, reg_alpha=0,
                             reg_lambda=1, scale_pos_weight=None, seed=1, subsample=1,
                             tree_method='exact', validate_parameters=1, verbosity=None), 'xgboost_w2v')

"""
Done training random_forest_w2v ...
              precision    recall  f1-score   support

           0       0.93      0.98      0.96      2706
           1       0.89      0.92      0.91      2669
           2       0.84      0.87      0.85      2690
           3       0.69      0.63      0.66      2749
           4       0.73      0.71      0.72      2767

    accuracy                           0.82     13581
   macro avg       0.82      0.82      0.82     13581
weighted avg       0.82      0.82      0.82     13581


Start training logistic_regression_w2v ...
Done training logistic_regression_w2v ...
              precision    recall  f1-score   support

           0       0.74      0.78      0.76      2706
           1       0.52      0.54      0.53      2669
           2       0.51      0.51      0.51      2690
           3       0.51      0.47      0.49      2749
           4       0.69      0.68      0.69      2767

    accuracy                           0.60     13581
   macro avg       0.59      0.60      0.60     13581
weighted avg       0.59      0.60      0.60     13581


Start training xgboost_w2v ...
[16:54:47] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
Done training xgboost_w2v ...
              precision    recall  f1-score   support

           0       0.81      0.88      0.85      2706
           1       0.65      0.68      0.67      2669
           2       0.62      0.60      0.61      2690
           3       0.55      0.47      0.50      2749
           4       0.67      0.70      0.68      2767

    accuracy                           0.67     13581
   macro avg       0.66      0.67      0.66     13581
weighted avg       0.66      0.67      0.66     13581

"""
