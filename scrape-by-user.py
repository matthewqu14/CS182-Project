import twint

# Saves a csv of all tweets from the past year (limited to 100) from users

with open("depressed_users.txt") as f:
    for user in f.readlines():
        filename = "depressed_tweets/" + user.strip("\n") + ".csv"
        # Configuration
        config = twint.Config()
        config.Username = user.strip("\n")
        config.Since = "2020-01-01"
        config.Limit = 100
        config.Store_csv = True
        config.Output = filename

        # Run
        twint.run.Search(config)
