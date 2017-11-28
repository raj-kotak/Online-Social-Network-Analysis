"""
classify.py
"""
from collections import Counter, defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import networkx as nx
import sys
import time
import os
import glob
import re
from scipy.sparse import csr_matrix
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np

def read_data(path):
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f, encoding='utf-8').readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f, encoding='utf-8').readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])

def tokenize(doc, keep_internal_punct=False):
    """
    Tokenize a string.
    The string should be converted to lowercase.
    If keep_internal_punct is False, then return only the alphanumerics (letters, numbers and underscore).
    If keep_internal_punct is True, then also retain punctuation that
    is inside of a word. E.g., in the example below, the token "isn't"
    is maintained when keep_internal_punct=True; otherwise, it is
    split into "isn" and "t" tokens.

    Params:
      doc....a string.
      keep_internal_punct...see above
    Returns:
      a numpy array containing the resulting tokens.

    >>> tokenize(" Hi there! Isn't this fun?", keep_internal_punct=False)
    array(['hi', 'there', 'isn', 't', 'this', 'fun'], 
          dtype='<U5')
    >>> tokenize("Hi there! Isn't this fun? ", keep_internal_punct=True)
    array(['hi', 'there', "isn't", 'this', 'fun'], 
          dtype='<U5')
    """
    ###TODO
    tokens_list = []
    if keep_internal_punct:
      tokens_list = np.array([re.sub('^\W+', '', re.sub('\W+$', '', x.lower())) for x in doc.split()])
    else:
      tokens_list = np.array(re.sub('\W+', ' ', doc.lower()).split())

    return tokens_list
    pass


def token_features(tokens, feats):
    """
    Add features for each token. The feature name
    is pre-pended with the string "token=".
    Note that the feats dict is modified in place,
    so there is no return value.

    Params:
      tokens...array of token strings from a document.
      feats....dict from feature name to frequency
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_features(['hi', 'there', 'hi'], feats)
    >>> sorted(feats.items())
    [('token=hi', 2), ('token=there', 1)]
    """
    ###TODO
    for token in tokens:
      feats["token="+token] += 1

    pass


def token_pair_features(tokens, feats, k=3):
    """
    Compute features indicating that two words occur near
    each other within a window of size k.

    For example [a, b, c, d] with k=3 will consider the
    windows: [a,b,c], [b,c,d]. In the first window,
    a_b, a_c, and b_c appear; in the second window,
    b_c, c_d, and b_d appear. This example is in the
    doctest below.
    Note that the order of the tokens in the feature name
    matches the order in which they appear in the document.
    (e.g., a__b, not b__a)

    Params:
      tokens....array of token strings from a document.
      feats.....a dict from feature to value
      k.........the window size (3 by default)
    Returns:
      nothing; feats is modified in place.

    >>> feats = defaultdict(lambda: 0)
    >>> token_pair_features(np.array(['a', 'b', 'c', 'd']), feats)
    >>> sorted(feats.items())
    [('token_pair=a__b', 1), ('token_pair=a__c', 1), ('token_pair=b__c', 2), ('token_pair=b__d', 1), ('token_pair=c__d', 1)]
    """
    ###TODO
    l = 0
    r = k
    while r <= len(tokens):
      temp = tokens[l:r]

      for i in range(0, len(temp)):
        for j in range(i+1, len(temp)):
          feats["token_pair="+temp[i]+"__"+temp[j]] += 1

         
      l = l + 1
      r = r + 1

    pass

f_pos = open('pos.txt', 'r+')
f_neg = open('neg.txt', 'r+')

pos_data = f_pos.readlines()
neg_data = f_neg.readlines()

pos_list = []
neg_list = []
for p in pos_data:
  pos_list.append(p[:-2])

for n in neg_data:
  neg_list.append(n[:-2])

neg_words = set(neg_list)
pos_words = set(pos_list)

def lexicon_features(tokens, feats):
    pos_counter = 0
    neg_counter = 0
    for token in tokens:
      if token.lower() in pos_words:
        pos_counter = pos_counter + 1
      
      if token.lower() in neg_words:
        neg_counter = neg_counter + 1

    feats['pos_words'] = pos_counter
    feats['neg_words'] = neg_counter 

    pass


def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for funct in feature_fns:
      funct(tokens, feats)

    return sorted(feats.items(), key=lambda x:(x[0]))
    pass


def vectorize(tokens_list, feature_fns, min_freq, vocab=None):
    row_list = []
    col_list = []
    data = []
    row = 0

    feats_list = []
    feats = defaultdict(lambda: 0)
    for tokens in tokens_list:
      feats = featurize(tokens, feature_fns)
      feats_list.append(dict(feats))
  
    if vocab == None:
      vocabulary = []
      visited = defaultdict(lambda: 0)
      freq = defaultdict(lambda: 0)
      vocab = defaultdict(lambda: 0)
      for feat in feats_list:
        for k, v in feat.items():
          if v > 0:
            freq[k] += 1
          if (k not in visited) and (freq[k] >= min_freq):
            vocabulary.append(k)
            visited[k] = 0

      vocabulary = sorted(vocabulary)
      count = 0
      for k in vocabulary:
        vocab[k] = count
        count += 1

    
    for feat in feats_list:
      for k, v in feat.items():
        if k in vocab:
          col_list.append(vocab[k])
          data.append(v)
          row_list.append(row)
      row += 1 

    X = csr_matrix((np.array(data,dtype='int64'), (np.array(row_list,dtype='int64'),np.array(col_list,dtype='int64'))), shape=(row, len(vocab)))
    return X, vocab
    pass


def accuracy_score(truth, predicted):
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(len(labels), k)
    accuracies = []
    for train_idx, test_idx in cv:
        clf.fit(X[train_idx], labels[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(labels[test_idx], predicted)
        accuracies.append(acc)
    avg = np.mean(accuracies)
    return avg
    pass


def eval_all_combinations(docs, labels, punct_vals,
                          feature_fns, min_freqs):
    result = []
    featureslist = []
    for f in range(1, len(feature_fns)+1):
          for features in combinations(feature_fns, f):
            featureslist.append(features)
    # print(featureslist)
    for punct in punct_vals:
      tokens_list = []
      for doc in docs:
        tokens_list.append(tokenize(doc, punct))
      for min_feq in min_freqs:
        for features in featureslist:
            features_list = list(features)
            X, vocab = vectorize(tokens_list, features_list, min_feq)
            result_dict = {}
            result_dict['punct'] = punct
            result_dict['features'] = features
            result_dict['min_freq'] = min_feq
            result_dict['accuracy'] = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
            result.append(result_dict)

    return sorted(result, key=lambda x: x['accuracy'], reverse=True)
    pass


def plot_sorted_accuracies(results):
    accuracy_list = [dicts['accuracy'] for dicts in results]
    plt.plot(range(len(accuracy_list)), sorted(accuracy_list), 'bo-')
    plt.xlabel('settings')
    plt.ylabel('accuracies')
    plt.savefig('accuracies.png')
    pass


def mean_accuracy_per_setting(results):
    accuracy_dict = defaultdict(list)
    for setting in results:
      if setting['features']:
        accuracy_dict['features='+' '.join([func.__name__ for func in list(setting['features'])])].append(setting['accuracy'])
      if setting['punct'] == True:
        accuracy_dict['punct=True'].append(setting['accuracy'])
      elif setting['punct'] == False:
        accuracy_dict['punct=False'].append(setting['accuracy'])
      if setting['min_freq'] == 2:
        accuracy_dict['min_freq=2'].append(setting['accuracy'])
      elif setting['min_freq'] == 5:
        accuracy_dict['min_freq=5'].append(setting['accuracy'])
      elif setting['min_freq'] == 10:
        accuracy_dict['min_freq=10'].append(setting['accuracy'])
      

    accuracy_setting_list = []
    for k, v in accuracy_dict.items():
      accuracy_setting_list.append((np.mean(v), k))

    return sorted(accuracy_setting_list, key=lambda x: x[0], reverse=True)  
    pass


def fit_best_classifier(docs, labels, best_result):
    tokens_list = [tokenize(doc, best_result['punct']) for doc in docs]

    X, vocab = vectorize(tokens_list, best_result['features'], best_result['min_freq'])

    clf = LogisticRegression()
    clf.fit(X, labels)

    return clf, vocab
    pass


def parse_test_data(best_result, vocab, tweets):
    test_tokens_list = [tokenize(test_doc, best_result['punct']) for test_doc in tweets]
    X_test, vocab_test = vectorize(test_tokens_list, best_result['features'], best_result['min_freq'], vocab)

    return X_test
    pass


def main():
    
    feature_fns = [token_features, token_pair_features, lexicon_features]
    docs, labels = read_data(os.path.join('data', 'train'))
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    clf, vocab = fit_best_classifier(docs, labels, results[0])
    best_result = results[0]

    f_test = open('tweets.txt', 'r+', encoding='utf-8')
    test_tweets = f_test.readlines()

    unique_tweets = set()
    for t in test_tweets:
        unique_tweets.add(t)
    unique_tweets = list(unique_tweets)

    X_test = parse_test_data(best_result, vocab, unique_tweets)
    predictions = clf.predict(X_test)
    
    pos_class_counter = 0
    neg_class_counter = 0
    pos_instances = []
    neg_instances = []
    for t in zip(predictions, unique_tweets):
        if t[0] == 1:
            pos_class_counter += 1
            pos_instances.append(t[1])
        elif t[0] == 0:
            neg_class_counter += 1
            neg_instances.append(t[1])
    
    f_classify = open('classifications.txt', 'w+', encoding='utf-8')
    
    f_classify.write('Number of positive instances found: '+str(pos_class_counter)+'\n')
    f_classify.write('Number of negative instances found: '+str(neg_class_counter)+'\n\n')

    f_classify.write('One example from each class:'+'\n')
    f_classify.write('Example for positive instance: '+pos_instances[0])
    f_classify.write('Example for negative instance: '+neg_instances[0])
    f_classify.close()
    

if __name__ == '__main__':
    main()
