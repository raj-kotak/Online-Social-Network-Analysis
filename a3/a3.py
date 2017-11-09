# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    movies['tokens'] = [tokenize_string(genre) for genre in movies.genres]
    return movies
    pass


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:

    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO
    genre_list = []
    for genres in movies['tokens']:
        for genre in genres:
            genre_list.append(genre)

    genre_dict = dict(Counter(genre_list))
    genre_list = sorted(genre_dict.keys())

    tokens_freq = defaultdict(lambda: 0)
    for index, row in movies.iterrows():
        tokens_freq[row.movieId] = dict(Counter(row.tokens))

    max_k = defaultdict(lambda: 0)
    for index, row in tokens_freq.items():
        max_k[index] = max(row.values())

    vocab = defaultdict(lambda: 0)
    vocab_counter = 0
    for genre in genre_list:
        vocab[genre] = vocab_counter
        vocab_counter += 1
    
    X_list = []
    for index, row in movies.iterrows():
        col_list = []
        row_list = []
        tfidf_list = []
        for genre in row.tokens:
            tfidf = (row.tokens.count(genre) / max_k[row.movieId] * math.log10((len(movies.movieId)) / genre_dict[genre]))
            if tfidf > 0:
                row_list.append(0)
                col_list.append(vocab[genre])
                tfidf_list.append(tfidf)

        X_list.append(csr_matrix((tfidf_list, (row_list, col_list)), shape=(1, len(vocab))))
    movies['features'] = X_list
    return movies, vocab
    pass

def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    return np.dot((a.toarray()[0]),(b.toarray()[0])) / ((np.sqrt(sum([i*i for i in a.toarray()[0]]))) * (np.sqrt(sum([i*i for i in b.toarray()[0]]))))
    pass

def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    prediction_list = []
    for test_index, test_row in ratings_test.iterrows():
        cosine_sum = 0.0
        weighted_sum = 0.0
        train_user_ratings = ratings_train[ratings_train['userId'] == test_row['userId']]
        movies_test_features = movies[movies['movieId'] == test_row['movieId']].iloc[0]['features']

        for train_index, train_row in train_user_ratings.iterrows():
            movies_train_features = movies[movies['movieId'] == train_row['movieId']].iloc[0]['features']
            cosine_val = cosine_sim(movies_test_features, movies_train_features)
            cosine_sum += cosine_val
            weighted_sum += cosine_val * train_row['rating']

        if cosine_sum != 0 and weighted_sum != 0:
            prediction_list.append(weighted_sum/cosine_sum)
        else:
            prediction_list.append(np.mean(train_user_ratings['rating']))
    
    return np.array(prediction_list)
    pass

def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def main():
    # download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
