# Project in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Authors: Anna Jonsson and Amanda Bigelius
# Start date: 2022-11-19
# Version: 1.0

# Heavily based on this tutorial: https://medium.com/@nikitasilaparasetty/twitter-sentiment-analysis-for-data-science-using-python-in-2022-6d5e43f6fa6e

# Importing libraries
import re                               # Regular expressions allows us to check if a specified string and a given regular expression match
import numpy as np                      # Working with arrays
import tweepy                           # Accessing the Twitter API
from tweepy import OAuthHandler         # Authentication handler
import matplotlib.pyplot as plt         # Visualize data in mulitple ways
import pandas as pd                     # Data manipulation and analysis
from textblob import TextBlob           # Process textual data for NLP, based on nltk
from wordcloud import WordCloud         # Create a word cloud to visualize the most common words
from better_profanity import profanity  # Remove profanity from tweets

# Importing our Twitter API keys
import API_KEYS

# Set the API-keys
consumer_key = API_KEYS.consumer_key()
consumer_secret = API_KEYS.consumer_secret()
access_token = API_KEYS.access_token()
access_token_secret = API_KEYS.access_token_secret()

# Access twitter data
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Ask user for input on what to search for
search_term = input("Enter search keyword: ")

# Remove retweets
search_term = search_term + " -filter:retweets"

# Get the latest 100 tweets on the topic
# HÄR KAN VI ÄNDRA SPRÅK, ANTAL TWEETS, OSV
tweets = tweepy.Cursor(api.search_tweets, q = search_term, lang = "en").items(100)

# Create a list of tweets, the users, and their location
list1 = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]

# Convert the list into a dataframe
df = pd.DataFrame(data=list1, 
                    columns=['tweets','user', "location"])

# Convert only the tweets into a list
tweet_list = df.tweets.to_list()

# Create a function to clean the tweets. Remove profanity, unnecessary characters, spaces, and stopwords.
def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    r = tweet.lower()
    r = profanity.censor(r)
    r = re.sub("'", "", r) # This is to avoid removing contractions in english
    r = re.sub("@[A-Za-z0-9_]+","", r)
    r = re.sub("#[A-Za-z0-9_]+","", r)
    r = re.sub(r'http\S+', '', r)
    r = re.sub('[()!?]', ' ', r)
    r = re.sub('\[.*?\]',' ', r)
    r = re.sub("[^a-z0-9]"," ", r)
    r = r.split()
    stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
    r = [w for w in r if not w in stopwords]
    r = " ".join(word for word in r)
    return r

# Run the list of tweets through the clean_tweet function and display the tweets
cleaned = [clean_tweet(tw) for tw in tweet_list]
cleaned

# Define the sentiment objects using TextBlob
sentiment_objects = [TextBlob(tweet) for tweet in cleaned]
sentiment_objects[0].polarity, sentiment_objects[0]

# Create a list of polarity values and tweet text
sentiment_values = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects]

# Print the value of the 0th row.
sentiment_values[0]

# Print all the sentiment values
sentiment_values[0:99]

# Create a dataframe of each tweet against its polarity
sentiment_df = pd.DataFrame(sentiment_values, columns=["polarity", "tweet"])
sentiment_df

# Save the polarity column as 'n'.
n=sentiment_df["polarity"]

# Convert this column into a series, 'm'. 
m=pd.Series(n)
m

# Initialize variables, 'pos', 'neg', 'neu'.

pos=0
neg=0
neu=0

# Create a loop to classify the tweets as Positive, Negative, or Neutral.
# Count the number of each.
for items in m:
    if items>0:
        print("Positive")
        pos=pos+1
    elif items<0:
        print("Negative")
        neg=neg+1
    else:
        print("Neutral")
        neu=neu+1
        
print(pos,neg,neu)

# Create a pie chart to visualize the results.
pieLabels=["Positive","Negative","Neutral"]
populationShare=[pos,neg,neu]
figureObject, axesObject = plt.subplots()
axesObject.pie(populationShare,labels=pieLabels,autopct='%1.2f',startangle=90)
axesObject.axis('equal')
plt.show()

# Display the number of twitter users who feel a certain way about the given topic.
print("%f percent of twitter users feel positive about %s"%(pos,search_term))
print("%f percent of twitter users feel negative about %s"%(neg,search_term))
print("%f percent of twitter users feel neutral about %s"%(neu,search_term))

# Create a Wordcloud from the tweets
all_words = ' '.join([text for text in cleaned])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
