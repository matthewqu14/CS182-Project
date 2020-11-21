# CS182-Project

File descriptions:
all_depressed_tweets.csv contains tweets with the word/hastag depression; used to find twitter accounts with positive
samples for depressed tweets

all_normal_tweets.csv contains negative samples of normal tweets

all_normal_vectors.csv is the vectorized version of tweets in all_normal_tweets

depressed_users.txt has a list of users reportedly diagonsed with depression, will use after training model to see the
threshold at which depressed users tweet out depressed tweets

sample_depressed_tweets.csv contains a sample of depressed (positive sample) tweets - will need to add more because
currently they come from only one user

sample_depressed_vectors.csv is the vectorized version of tweets in sample_depressed_tweets

users_for_depressed_tweets.txt contains twitter users that have a lot of depressed tweets for positive samples, generated
by manually going through all_depressed_tweets and looking at profiles.
