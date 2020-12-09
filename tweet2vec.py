import nltk
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import re
import os
import contractions
import ftfy
import spacy
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# setup command - run only once
# nltk.download('stopwords')

# create new directory to store output
outdir = "./vectors"
if not os.path.exists(outdir):
    os.mkdir(outdir)

# download model using command "python -m spacy download en_core_web_sm"
nlp = spacy.load("en_core_web_sm")

files = ["all_normal_tweets.csv", "all_depressed_tweets.csv"]
dimensions = [25, 50, 100, 200]
word_dict = {}

for i in range(1, 2):  # iterate through files, using index i as classification of tweet
    for VECTOR_DIMENSION in [25]:
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

        # create DataFrame
        tweets = pd.read_csv(FILE)["tweet"]
        all_vectors = pd.DataFrame()

        for tweet in tweets:
            # create numpy vector for storing vector of tweet
            vector = np.zeros(VECTOR_DIMENSION, dtype=np.float64)
            word_count = 0

            # remove @mentions, hastags, links, and emojis
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

            # remove common words (stop words) and lemmatize words
            words = nlp(tweet)
            words = [word.lemma_ for word in words if word.lemma_ not in stopwords.words("english")]

            for word in words:
                if word in model.vocab:
                    word_dict[word] = word_dict.get(word, 0) + 1
                    vector += model[word]
                    word_count += 1

            if np.any(vector):
                vector /= word_count
                vector = np.append(vector, i)
                all_vectors = all_vectors.append(pd.DataFrame(vector).T, ignore_index=True)

        all_vectors.rename(columns={VECTOR_DIMENSION: "Classification"}, inplace=True)

        all_vectors.to_csv(os.path.join(outdir, FILE.replace("tweets", f"vectors_{VECTOR_DIMENSION}")))

# creating word cloud
# extra = ["cq", "tg", "mc", "tj", "ga", "kik", "rc", "jm", "ifb"]
# lst = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
# d = dict(lst)
# for k in extra:
#     if k in d:
#         del d[k]
#
# wc = WordCloud(colormap="winter", background_color="white", width=1500, height=1000, max_words=100).\
#     generate_from_frequencies(d)
# plt.figure(figsize=(12, 8))
# plt.imshow(wc)
# plt.axis("off")
# plt.show()
