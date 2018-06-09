
# PySentiments
------------------
#### Analysis
- At the time of retrieving and analyzing the data, @BBCWorld appeared to have the most negative tweets, while @CNN appeared to have the most positive tweets.
- Of the five news sources analyzed, @nytimes seemed to have the most neutral tweets although they do lean positive.
- Given the spread in the scatterplot, @BBCWorld appears to tweet at a lower frequency compared to the others.


```python
# Dependencies
import tweepy
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

# Twitter API Keys
from config import consumer_key, consumer_secret, access_token, access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Function to analyze tweets; import twitter username, output plot
def AnalyzeTweets(target_users):
    counter = 1
    data = []
    oldest_tweet = None
    
    for target_user in target_users:
        # loop through 100 tweets (5 pages)
        for i in range(5):
            # Get tweets
            public_tweets = api.user_timeline(target_user) #, max_id = oldest_tweet)

            for tweet in public_tweets:
                #print("Tweet {}: {} - {}".format(counter, tweet["text"], tweet["created_at"]))

                # Get relative timestamp from time format "Sat May 26 18:04:07 +0000 2018"
                created_at = tweet["created_at"]
                timestamp = datetime.strptime(created_at, "%a %b %d %H:%M:%S %z %Y").timestamp()

                # Analyze each tweet
                results = analyzer.polarity_scores(tweet["text"])
                compound = results["compound"]

                data.append({"Target User": target_user, 
                                  "Timestamp": timestamp,
                                  "Compound": compound,
                                  "Tweets Ago": counter})

                # Increment counter
                counter += 1
            
    # Turn data into dataframe
    df = pd.DataFrame(data)
    
    # Plot data
    for target, group in df.groupby("Target User"):
        plt.scatter(x=group["Timestamp"], y=group["Compound"], label=target)
    
    plt.legend(bbox_to_anchor=(1, 1))
    plt.grid(alpha=0.3)
    plt.ylim(-1,1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    plt.title("Sentiment Analysis of Tweets ({})".format(now))
    plt.xlabel("Timestamp")
    plt.ylabel("Compound Score")
    
    return df

```


```python
# List of target users to analyze 
target_users = ["@BBCWorld", "@CBSNews", "@CNN", "@FoxNews", "@nytimes"]
```


```python
# Call function to retrieve and analyze tweet sentiments
df = AnalyzeTweets(target_users)
plt.show()
```


![png](output_4_0.png)



```python
avg_sentiments = df.groupby("Target User")["Compound"].mean()
avg_sentiments
```




    Target User
    @BBCWorld   -0.030280
    @CBSNews    -0.019875
    @CNN         0.008405
    @FoxNews     0.031130
    @nytimes     0.005840
    Name: Compound, dtype: float64




```python
# Make bar plot with overall tweet sentiments
x_axis = np.arange(0, len(target_users))
tick_locations = []
for x in x_axis:
    tick_locations.append(x)

plt.bar(x_axis, avg_sentiments)
plt.xticks(tick_locations, target_users)
plt.grid(alpha=0.2)
plt.axhline(0, color='black', alpha=1, lw=0.5)

for i, v in enumerate(avg_sentiments):
    plt.text(i-0.25, (v/2), "{:.3f}".format(v), color='white')

plt.title("Overall Tweet Sentiments")
plt.ylabel("Average Compound Sentiment Score")

plt.show()
```


![png](output_6_0.png)

