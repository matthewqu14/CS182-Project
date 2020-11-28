import nltk
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import re
import os
import contractions
import ftfy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# setup command - run only once
# nltk.download('stopwords')

# create new directory to store output
outdir = "./vectors"
if not os.path.exists(outdir):
    os.mkdir(outdir)

files = ["all_normal_tweets.csv", "all_depressed_tweets.csv"]
dimensions = [25, 50, 100, 200]

for i in range(2):  # iterate through files, using index i as classification of tweet
    for VECTOR_DIMENSION in dimensions:
        MODEL_NAME = "glove-twitter-" + str(VECTOR_DIMENSION)
        FILE = files[i]

        # load model using API if not downloaded locally
        # path = api.load(MODEL_NAME, return_path=True)
        # model = KeyedVectors.load_word2vec_format(path)

        # save model locally
        # model.init_sims(replace=True)
        # model.save(f"models/{MODEL_NAME}.model")

        # load model if downloaded locally
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
                vector = np.append(vector, i)
                all_vectors = all_vectors.append(pd.DataFrame(vector).T, ignore_index=True)

        all_vectors.rename(columns={VECTOR_DIMENSION: "Classification"}, inplace=True)

        all_vectors.to_csv(os.path.join(outdir, FILE.replace("tweets", f"vectors_{VECTOR_DIMENSION}")))
