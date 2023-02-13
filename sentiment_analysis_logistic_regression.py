# Project in the course TNM108 - Machine Learning for Social Media at Linköpings University, Fall 2022
# Authors: Anna Jonsson and Amanda Bigelius
# Start date: 2023-01-23
# Version: 2.0

# Follows this tutorial: https://www.kaggle.com/code/katearb/sentiment-analysis-in-twitter-93-test-acc 

# Import libraries
import numpy as np
import pandas as pd
import pickle

# EDA - Exploratory Data Analysis
import matplotlib.pyplot as plt
import seaborn as sns

# Top words
# import re     # NOT same as regex!! Might create bug!
import nltk
from nltk.corpus import stopwords

# Correlated words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

# ANOVA test to ensure the distrubution of tweets lengths doesnt differ
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Emoticons
import emoji
import regex as re      # Again, NOT same as ordinary 're'!

# Preprocessing
from tqdm import tqdm
# NLP processing
import spacy
from spacy.lang.en import English

# Data processing and analytics
from sklearn.preprocessing import OneHotEncoder         # Categorical --> binary
#from sklearn.utils.validation import check_is_fitted
#from sklearn.model_selection import train_test_split
#from sklearn.exceptions import NotFittedError

# Training
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from sklearn.model_selection import  KFold
from sklearn.metrics import confusion_matrix

# MI feature selection
from sklearn.feature_selection import mutual_info_classif as MIC

import warnings
warnings.filterwarnings("ignore")

# Load the datasets - data & validation data
data = pd.read_csv('dataset/twitter_training.csv', header=None)
val_data = pd.read_csv('dataset/twitter_validation.csv', header=None)

# Rename the columns
data.columns = ['tweet_id', 'subject', 'sentiment', 'text']
val_data.columns = ['tweet_id', 'subject', 'sentiment', 'text']

# Global names - dependent variable
TARGET = 'sentiment'

# *************************************************************#
##          EDA - Exploratory Data Analysis                   ##
# *************************************************************#

## Show how the data works
# data.info()

## Non valid data values ("Nans")   - Not necessary while testing
# data.isnull().sum()
# val_data.isnull().sum()

# Drop samples with nans/ missing values
data.dropna(inplace = True, axis = 0)
# val_data.dropna(inplace = True, axis = 0)     # There is no missing values in the validation data set, we don't need to drop any rows

# Text stats
texts = data['text']
text_lens = [len(t.split()) for t in texts.values]
len_mean = np.mean(text_lens)

# Plot the distrubution of the lengths of the tweets
fig, axes = plt.subplots(2, 1, figsize=(15, 8))
axes[0].set_title('Distribution of nuber of tokens in tweets')
sns.boxplot(text_lens, ax = axes[0])
sns.histplot(text_lens, bins = 100, kde = True, ax = axes[1])
axes[1].vlines(len_mean, 0, 5000, color = 'red')
plt.annotate('mean', xy = (len_mean, 5000), xytext = (len_mean - 2, 5050), color = 'r')
plt.show()

# Find and examine extreme outliers
extreme_outliers = data['text'][np.array(text_lens) > 125]

# for idx in extreme_outliers.index:
#    print(idx, 'Target', data[TARGET][idx])
#    print(extreme_outliers[idx])
#    print('=-=-=-=-=-=-=-=-'*4, '\n')

# Investigate outliers closer to the majority
outliers = data['text'][np.array(text_lens) > 60]

# for idx in outliers.index:
#    print(idx, 'Target', data[TARGET][idx])
#    print(outliers[idx])
#    print('=-=-=-=-=-=-=-=-'*4, '\n')

## Target Analysis ##

# Balance
target_balance = data[TARGET].value_counts()

# Plot the balance
plt.figure(figsize=(5, 5))
plt.pie(target_balance, labels=[f'{idx}\n{round(target_balance[idx]/len(data), 2)}' for idx in target_balance.index], 
        colors=['r', '#00FF00', '#FFFF00', 'gray'])
plt.title('Proportions of target classes')
plt.show()

## Find the most common words in the tweets ##
stopwords_list = stopwords.words('english')

word_counts = {'Positive': [], 'Neutral': [], 'Irrelevant': [], 'Negative': []}

pattern = re.compile('[^\w ]')

# Remove stopwords, make lowercase, and split into words
for text, t in zip(data['text'], data[TARGET]):
    text = re.sub(pattern, '', text).lower().split()
    text = [word for word in text if word not in stopwords_list]
    word_counts[t].extend(text)

# Plot the most common words
fig, axes = plt.subplots(2, 2, figsize=(20,10.5))
for axis, (target, words) in zip(axes.flatten(), word_counts.items()):
    bar_info = pd.Series(words).value_counts()[:25]
    sns.barplot(x=bar_info.values, y=bar_info.index, ax=axis)
    axis.set_title(f'Top words for {target}')
plt.show()

## Most correlated words for each topic using chi2 ##
tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, ngram_range = (1, 2), stop_words = 'english', max_features = 10000)
features = tfidf.fit_transform(data['text']).toarray()
labels = data[TARGET]

print("Each of the %d Text is represented by %d features (TF-IDF score of unigrams and bigrams)" %(features.shape))

N = 10
for label in set(labels):
    features_chi2 = chi2(features, labels == label)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("\n==> %s:" %(label))
    print("  * Most Correlated Unigrams are: %s" %(', '.join(unigrams[-N:])))
    print("  * Most Correlated Bigrams are: %s" %(', '.join(bigrams[-N:])))

# Length for classes without outliers
tweets_len = {'Positive': [], 'Neutral': [], 'Irrelevant': [], 'Negative': []}
pattern = re.compile('[^\w ]')
tweets_len = pd.DataFrame([len(re.sub(pattern, '', text).lower().split()) for text in data['text'] if len(text) < 125], columns = ['len'])
tweets_len['target'] = data[TARGET]

# Plot the distribution of the lengths of the tweets
plt.figure(figsize=(20, 7))
sns.kdeplot(data=tweets_len, x='len', hue='target')
plt.show()

# Check equal disposition, and normal-like distribution, using ANOVA (Analysis of Variance) test
# Perform two-way ANOVA
model = ols('len ~ target', data=tweets_len).fit()
sm.stats.anova_lm(model, typ=2)

## Emojis ##
def split_count(text):
    emoji_list = []
    data = re.findall(r'\X', text)
    for word in data:
        if any(char in emoji.EMOJI_DATA for char in word):
            emoji_list.append(word)
    
    return emoji_list

target_emojis = {'Positive': [], 'Neutral': [], 'Irrelevant': [], 'Negative': []}

pattern = re.compile('\u200d')
for i, text in enumerate(texts):
    emoji_count = split_count(text)
    if emoji_count:
        emoji_count = [re.sub(pattern, '', e) for e in emoji_count]
        target_emojis[data[TARGET].iloc[i]].extend(emoji_count)

fig, axes = plt.subplots(2, 2, figsize=(20,10.5))
for t, emojis in target_emojis.items():
    plt.figure(figsize=(10, 5))
    bar_info = pd.Series(emojis).value_counts()[:20]
    print('=========='*10,  f'\nTop emojis for {t} \n', list(bar_info.index))
    bar_info.index = [emoji.demojize(i, delimiters=("", "")) for i in bar_info.index]
    sns.barplot(x=bar_info.values, y=bar_info.index)
        
    plt.title(f'{t}')
    plt.show()

# **********************************************************#
#               START OF PROCESSING DATA                    #
# **********************************************************#

### Preprocessing ###

capitalized = [np.sum([t.isupper() for t in text.split()]) for text in np.array(data['text'])]
# capitalized_target = pd.DataFrame([(c, t) for c, t in zip(capitalized, data[TARGET])], columns=['cap', 'target'])
# capitalized_target_no_outliers = capitalized_target[capitalized_target['cap'] < 75]

# Why ??
ids_to_remove = [1826, 10454, 32186, 68078]
data = data[~data.index.isin(ids_to_remove)]
data.index = range(len(data))

# Collects all the cleanup methods into a class. 
# Put smaller cleanup functions into 'transform' function.
# Calls the cleanup by pr = Preprocessor(), and then pr.transform(...)

nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

class Preprocessor:
    
    #Initialises class variables 'vectorizer', 'stopwords', vectorizer_fitted'
    #OBS: max_features here should match init of TFIDF higher up in code, or at least be lower?
    def __init__(self, stopwords = stopwords):
        self.vectorizer = TfidfVectorizer(lowercase = False, max_features = 10000,
                                         min_df = 10, ngram_range=(1, 3),
                                         tokenizer=None)
        self.stopwords = stopwords
        self.vectorizer_fitted = False
        
    def remove_urls(self, texts):
        print('Removing URLs...')
        pattern = re.compile('(\w+\.com ?/ ?.+)|(http\S+)')
        return [re.sub(pattern, '', text) for text in texts]
    
    def remove_double_space(self, texts):
        print('Removing double space...')
        pattern = re.compile(' +')
        return [re.sub(pattern, ' ', text) for text in texts]
        
    def remove_punctuation(self, texts):
        print('Removing Punctuation...')
        pattern = re.compile('[^a-z ]')
        return [re.sub(pattern, ' ', text) for text in texts]
    
    def remove_stopwords(self, texts):
        print('Removing stopwords...')
        return [[w for w in text.split(' ') if w not in self.stopwords] for text in tqdm(texts)]
    
    def remove_numbers(self, texts):
        print('Removing numbers...')
        return [' '.join([w for w in text if not w.isdigit()]) for text in tqdm(texts)]
    
    def decode_emojis(self, texts):
        print('Decoding emojis...')
        return [emoji.demojize(text, language='en') for text in texts] 
    
    def lemmatize(self, texts):
        print('Lemmatizing...')
        lemmatized_texts = []
        for text in tqdm(texts):
            doc = nlp(text)
            lemmatized_texts.append(' '.join([token.lemma_ for token in doc]))
                                    
        return lemmatized_texts
        
    def transform(self, X, y=None, mode='train'):
        X = X.copy()
        print('Removing Nans...')
        X = X[~X.isnull()]                          # delete nans
        X = X[~X.duplicated()]                      # delete duplicates
        
        if mode == 'train':
            self.train_idx = X.index
        else:
            self.test_idx = X.index
        
        #We dont use 'cap' after this. Still necessary?
        print('Counting capitalized...')
        capitalized = [np.sum([t.isupper() for t in text.split()]) 
                           for text in np.array(X.values)]  # count capitalized
        # X['cap'] = capitalized
        print('Lowering...')
        X = [text.lower() for text in X]             # lower
        X = self.remove_urls(X)                      # remove urls
        X = self.remove_punctuation(X)               # remove punctuation
        X = self.remove_double_space(X)              # remove double space
        X = self.decode_emojis(X)                    # decode emojis
        X = self.remove_stopwords(X)                 # remove stopwords
        X = self.remove_numbers(X)                   # remove numbers                      
        X = self.lemmatize(X)                        # lemmatize
        
        #Checks if we've already fitted the data
        # --> Same result no matter how many times we use pr.transform(...)
        if not self.vectorizer_fitted:
            self.vectorizer_fitted = True
            print('Fitting vectorizer...')
            self.vectorizer.fit(X)

        print('Vectorizing...')
        X = self.vectorizer.transform(X)             # vectorize
        
        return X
    
pr = Preprocessor()

data_train = data
data_test = val_data

y_train = data['sentiment']
y_test = val_data['sentiment']

# Cleanup --> processed data
data_train_pr = pr.transform(data_train['text'])

# Makes sparse matrix out of the cleaned relevant words (?)
data_train_pr = pd.DataFrame.sparse.from_spmatrix(data_train_pr, columns=pr.vectorizer.get_feature_names_out())

# Convert categorys to binary values. 
# If word is in the category: 1
# If word is not in category: -1
print('OHE...')
ohe = OneHotEncoder()
referring_ohe = ohe.fit_transform(data_train['subject'][data_train.index.isin(pr.train_idx)].to_numpy().reshape(-1, 1))
referring_ohe = pd.DataFrame.sparse.from_spmatrix(referring_ohe, columns=ohe.get_feature_names_out())

# Concat mashes the sparse matrix into a single string / array?
X_train = pd.concat([data_train_pr, referring_ohe], axis=1)
y_train = y_train[y_train.index.isin(pr.train_idx)]
y_train.index = X_train.index

data_test_pr = pr.transform(data_test['text'], mode='test')
data_test_pr = pd.DataFrame.sparse.from_spmatrix(data_test_pr, columns=pr.vectorizer.get_feature_names_out())

print('OHE again...')
ohe = OneHotEncoder()
referring_ohe = ohe.fit_transform(data_train['subject'][data_train.index.isin(pr.train_idx)].to_numpy().reshape(-1, 1))
referring_ohe = ohe.transform(data_test['subject'][data_test.index.isin(pr.test_idx)].to_numpy().reshape(-1, 1))
referring_ohe = pd.DataFrame.sparse.from_spmatrix(referring_ohe, columns=ohe.get_feature_names_out())

X_test = pd.concat([data_test_pr, referring_ohe], axis=1)
y_test = y_test[y_test.index.isin(pr.test_idx)]
y_test.index = X_test.index

print('Shape of X_test and y_test...')
X_test.shape, y_test.shape

# Save prepared data for future use
with open('../X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('../X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)

with open('../y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)
with open('../y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)

### Training ###

# Cross validation, gridsearch
def train_cv(model, X_train, y_train, params, n_splits = 5, scoring='f1_weighted'):
    kf = KFold(n_splits = n_splits, random_state = 0, shuffle = True)

    cv = RandomizedSearchCV(model, params, cv = kf, scoring = scoring, return_train_score = True, n_jobs = -1, verbose = 2, random_state = 1)
    cv.fit(X_train, y_train)

    print('Best params', cv.best_params_)
    return cv

rs_parameters = {
    'penalty': ['l2', 'l1', 'elasticnet'],
    'C': uniform(scale = 10),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],
    'l1_ratio': uniform(scale = 10)
    }

### Training without feature selection ###
lr = LogisticRegression()

print('Cross Validation...')
model_cv_lr = train_cv(lr, X_train, y_train, rs_parameters)

print('Best estimator...')
bestimator_lr = model_cv_lr.best_estimator_

print('Classification report...')
print(classification_report(y_test, bestimator_lr.predict(X_test)))

# Confusion matrix
# sns.heatmap(confusion_matrix(y_test, bestimator_lr.predict(X_test)), annot=True)
# plt.show()

### With MI Feature Selection ###
print('Mutual Info Classification...')
mi_score = MIC(X_train,y_train)

cols_importance = sorted(list(zip(X_train.columns, mi_score)), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 8))
mi_imp = [pair[1] for pair in cols_importance[:30]]
cols = [pair[0] for pair in cols_importance[:30]]
sns.barplot(x=mi_imp, y=cols)
plt.title('The most important features')
plt.show()

plt.figure(figsize=(10, 8))
mi_imp = [pair[1] for pair in cols_importance[-30:]]
cols = [pair[0] for pair in cols_importance[-30:]]
sns.barplot(x=mi_imp, y=cols)
plt.title('The least important features')
plt.show()

X_train_6k = X_train[[pair[0] for pair in cols_importance[:6000]]]
X_test_6k = X_test[[pair[0] for pair in cols_importance[:6000]]]

# Save prepared data for future use
# with open('../X_train_6k.pkl', 'wb') as f: pickle.dump(X_train_6k, f)
# with open('../X_test_6k.pkl', 'wb') as f: pickle.dump(X_test_6k, f)
    
# with open('../X_train_6k.pkl', 'rb') as f: X_train_6k = pickle.load(f)
# with open('../X_test_6k.pkl', 'rb') as f: X_test_6k = pickle.load(f)

# Leave 6k features
lr = LogisticRegression()
print('Cross Validation FS 6k...')
model_cv_lr_6k = train_cv(lr, X_train_6k, y_train, rs_parameters)

print('Best estimator FS 6k...')
bestimator_lr_6k = model_cv_lr_6k.best_estimator_

print('Classification report FS 6k...')
print(classification_report(y_test, bestimator_lr_6k.predict(X_test_6k)))

# sns.heatmap(confusion_matrix(y_test, bestimator_lr_6k.predict(X_test_6k)), annot=True)
# plt.show()

print('- - - - - DONE - - - - - ')

# Binary Classification + TextBlob Sentiment Analysis 

# Neural networks