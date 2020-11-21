import nltk
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import contractions
import ftfy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# choose between 25, 50, 100, 200
VECTOR_DIMENSION = 50
MODEL_NAME = "glove-twitter-" + str(VECTOR_DIMENSION)
FILE = "all_normal_tweets.csv"

# Setup commands
# nltk.download('stopwords')
# path = api.load(MODEL_NAME, return_path=True)
# model = KeyedVectors.load_word2vec_format(path)
# model.init_sims(replace=True)
# model.save(f"models/{MODEL_NAME}.model")

# load the model
model = KeyedVectors.load(f"models/{MODEL_NAME}.model", mmap="r")

# create dataframe
tweets = pd.read_csv(FILE)["tweet"]
all_vectors = pd.DataFrame()

for tweet in tweets:
    # create numpy vector for storing vector of tweet
    vector = np.zeros(VECTOR_DIMENSION, dtype=np.float64)
    word_count = 0

    # remove @mentions, hastags, links, emojis, and punctuation
    tweet = re.sub("(@[\w]*)|(#[\w]*)|(http\S+)", "", tweet)
    tweet = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE).sub('', tweet)
    tweet = ftfy.fix_text(tweet)

    # check that tweet is properly cleaned
    # if re.search("@+|#+|(http)+", tweet):
    #     print(tweet)
    #     exit(1)

    # expand contractions and remove punctuation
    tweet = contractions.fix(tweet)
    tweet = re.sub("[^\w\s]", "", tweet).lower()

    # remove common words (stop words)
    words = word_tokenize(tweet)
    words = [word for word in words if word not in stopwords.words("english")]

    for word in words:
        if word in model.vocab:
            vector += model[word]
            word_count += 1

    if np.any(vector):
        vector /= word_count
        vector = np.append(vector, 1)
        all_vectors = all_vectors.append(pd.DataFrame(vector).T, ignore_index=True)

all_vectors.rename(columns={50: "Classification"}, inplace=True)

all_vectors.to_csv(FILE.replace("tweets", "vectors"))