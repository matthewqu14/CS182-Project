import twint

# Scrape for normal (non-depressive) tweets

# Configuration
config = twint.Config()
config.Search = "I"
config.Since = "2019-11-18"
config.Limit = 10000
config.Store_csv = True
config.Output = "all_normal_tweets.csv"

# Run
twint.run.Search(config)
