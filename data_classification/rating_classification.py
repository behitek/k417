import pickle
import re
import string

import pandas as pd
import seaborn as sns  # used for plot interactive graph.
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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
    return _row


def remake_label(row):
    # return row['Rating'] - 1
    if row['Rating'] <= 3:
        return 0
    return 1


data['Review'] = data.apply(lambda row: preprocess(row), axis=1)
data['Rating'] = data.apply(lambda row: remake_label(row), axis=1)

X = data['Review']
y = data['Rating']

print('Done pre processing')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def train_pipeline(classifier, name='default'):
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1),
                                                  max_df=0.8,
                                                  max_features=30000)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', classifier)
                         ])
    print('Start training {} ...'.format(name))
    text_clf = text_clf.fit(X_train, y_train)
    print('Done training {} ...'.format(name))
    y_pred = text_clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Save model
    pickle.dump(text_clf, open("{}.pkl".format(name), "wb"))
    print()


train_pipeline(RandomForestClassifier(max_depth=None), 'random_forest')
train_pipeline(LogisticRegression(max_iter=10000), 'logistic_regression')
train_pipeline(XGBClassifier(use_label_encoder=False, base_score=0.5, booster='gbtree', colsample_bylevel=1,
                             colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                             importance_type='gain', interaction_constraints='',
                             learning_rate=0.05, max_delta_step=0, max_depth=None,
                             min_child_weight=1, monotone_constraints='()',
                             n_estimators=100, n_jobs=0, num_parallel_tree=1,
                             objective='multi:softprob', random_state=1, reg_alpha=0,
                             reg_lambda=1, scale_pos_weight=None, seed=1, subsample=1,
                             tree_method='exact', validate_parameters=1, verbosity=None), 'xgboost')

"""
Done pre processing
Start training random_forest ...
Done training random_forest ...
              precision    recall  f1-score   support

           0       0.80      0.39      0.53       292
           1       0.28      0.03      0.05       333
           2       0.60      0.01      0.01       432
           3       0.41      0.26      0.32      1252
           4       0.54      0.94      0.68      1790

    accuracy                           0.52      4099
   macro avg       0.53      0.33      0.32      4099
weighted avg       0.50      0.52      0.44      4099


Start training logistic_regression ...
Done training logistic_regression ...
              precision    recall  f1-score   support

           0       0.80      0.61      0.69       292
           1       0.49      0.41      0.45       333
           2       0.44      0.24      0.31       432
           3       0.55      0.51      0.53      1252
           4       0.70      0.86      0.77      1790

    accuracy                           0.63      4099
   macro avg       0.60      0.53      0.55      4099
weighted avg       0.62      0.63      0.62      4099


Start training xgboost ...
[14:52:19] WARNING: ../src/learner.cc:1115: Starting in XGBoost 1.3.0, the default evaluation metric used with the objective 'multi:softprob' was changed from 'merror' to 'mlogloss'. Explicitly set eval_metric if you'd like to restore the old behavior.
Done training xgboost ...
              precision    recall  f1-score   support

           0       0.71      0.49      0.58       292
           1       0.39      0.17      0.23       333
           2       0.50      0.13      0.21       432
           3       0.47      0.39      0.43      1252
           4       0.60      0.88      0.72      1790

    accuracy                           0.56      4099
   macro avg       0.54      0.41      0.43      4099
weighted avg       0.54      0.56      0.52      4099
"""
