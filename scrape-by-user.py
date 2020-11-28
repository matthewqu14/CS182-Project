import twint

# Scrape for depressive tweets

with open("users_for_depressed_tweets.txt") as f:
    for user in f.readlines():
        # Configuration
        config = twint.Config()
        config.Username = user.strip("\n")
        config.Store_csv = True
        config.Output = "all_depressed_tweets.csv"

        # Run
        twint.run.Search(config)
