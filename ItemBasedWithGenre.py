# coding: utf-8
#  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/

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
    movies['tokens'] = pd.Series([tokenize_string(i) for i in movies['genres']]).values
    return movies



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
    index = 0
    mat_list = []
    vocab = {}
    df_i = defaultdict(int)

    for genres in movies["tokens"].iteritems():
        for genre in set(genres[1]):
            df_i[genre] += 1

    for key,value in sorted(df_i.items(), key=lambda x: x[0]):
        vocab[key] = index
        index += 1


    for genres in movies["tokens"].iteritems():
        c = Counter(genres[1])
        data = []
        row = []
        coloumn = []
        for genre in genres[1]:
            row.append(0)
            coloumn.append(vocab[genre])
            data.append((genres[1].count(genre)/c.most_common(1)[0][1]) * math.log10(movies.shape[0]/df_i[genre]))

        mat_list.append(csr_matrix((data,(row,coloumn)),shape=(1,len(vocab))))

    movies['features'] = pd.Series(mat_list).values

    return (movies,vocab)



def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::5])
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
    a = a.todense()
    b = b.todense()
    eucledian_prod =  np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b)))
    return (np.dot(a, b.T).sum()) / eucledian_prod



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
    user_rating = defaultdict(lambda : [[],0,0])
    predictions = []

    for index,row in ratings_train.iterrows():
        user_rating[row["userId"]][0].append((row["movieId"],row["rating"]))
        user_rating[row["userId"]][1] += row["rating"]
        user_rating[row["userId"]][2] = user_rating[row["userId"]][1] / len(user_rating[row["userId"]][0])

    for index, row in ratings_test.iterrows():
        a = movies.loc[movies["movieId"] == row["movieId"],"features"].iloc[0]
        average = user_rating[row["userId"]][2]
        stats = [0,0]
        for movies_info in user_rating[row["userId"]][0]:
            b = movies.loc[movies["movieId"] == movies_info[0],"features"].iloc[0]
            similarity = cosine_sim(a,b)
            i = 0
            if(similarity >0):
                stats[0] += similarity
                stats[1] += similarity * movies_info[1]
                i += 1
            if(i>0):
                average = (stats[1]/stats[0])
        predictions.append(average)

    return np.array(predictions)



def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()

def rmse(predictions, ratings_test):
    return np.sqrt(((predictions - ratings_test.rating) ** 2).mean())

def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % rmse(predictions, ratings_test))


if __name__ == '__main__':
    main()
