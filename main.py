# Project in the course TNM108 - Machine Learning for Social Media at Linköpings University 2022
# Authors: Anna Jonsson and Amanda Bigelius
# Start date: 2022-11-19
# Version: 1.0

# Heavily based on this tutorial: https://medium.com/@nikitasilaparasetty/twitter-sentiment-analysis-for-data-science-using-python-in-2022-6d5e43f6fa6e

# Importing libraries
import re                                   # Regular expressions allows us to check if a specified string and a given regular expression match
import numpy as np                          # Working with arrays
import tweepy                               # Accessing the Twitter API
from tweepy import OAuthHandler             # Authentication handler
import matplotlib.pyplot as plt             # Visualize data in mulitple ways
import pandas as pd                         # Data manipulation and analysis
from textblob import TextBlob               # Process textual data for NLP, based on nltk
from wordcloud import WordCloud             # Create a word cloud to visualize the most common words
from better_profanity import profanity      # Remove profanity from tweets
from nltk.corpus import stopwords           # Remove stopwords from the tweets
from collections import Counter             # Count the number of times a value appears in a list

# Importing libraries for the GUI
import PySimpleGUI as sg                    # To make the GUI  

# Importing our Twitter API keys from API_KEYS.py
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

# Create GUI
layout = [[sg.Text("Enter a search keyword: "), sg.Input(key = 'query')],
          [sg.Button("Search"), sg.Button("Exit")]]

# Create window for GUI
window = sg.Window("Twitter Sentiment Analysis", layout)

# Create empty string to declare the search term
query = ''

# Create an event loop, exits when the user clicks the OK or exit button
while True:
    event, values = window.read()
    # End program if user closes window or clicks Exit
    if event == "Exit" or event == sg.WIN_CLOSED:
        print('User exited the program')
        quit()
    # If the user clicks the search button, the program will run
    if event == "Search":
        # Define search term from input
        query = values['query']
        print('User searched for ' + query)
        break
window.close()

# Continue with the Twitter Sentiment Analysis

# Remove retweets
search_term = query + " -filter:retweets"

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
    if type(tweet) == np.float: # If the tweet is a float, return an empty string
        return ""
    r = tweet.lower() # Convert the tweet to lowercase
    r = profanity.censor(r) # Remove profanity
    r = re.sub("'", "", r) # This is to avoid removing contractions in english
    r = re.sub("@[A-Za-z0-9_]+","", r)
    r = re.sub("#[A-Za-z0-9_]+","", r)
    r = re.sub(r'http\S+', '', r)
    r = re.sub('[()!?]', ' ', r)
    r = re.sub('\[.*?\]',' ', r)
    r = re.sub("[^a-z0-9]"," ", r)
    r = r.split()

    # Declare stopwords to remove from the tweets
    stop_words = set(stopwords.words('english'))

    # Remove stopwords from the tweets
    r = [w for w in r if not w in stop_words]

    # Join the words back together
    r = " ".join(word for word in r)
    return r

# Run the list of tweets through the clean_tweet function and display the tweets
cleaned = [clean_tweet(tw) for tw in tweet_list]
cleaned

# Define the sentiment objects using TextBlob
# TODO: Change the function so it uses Flair/ NLTK instead or another vector based function
sentiment_objects = [TextBlob(tweet) for tweet in cleaned]
sentiment_objects[0].polarity, sentiment_objects[0]

# Create a list of polarity values and tweet text
# TODO: Change the function so it uses Flair/ NLTK instead or another vector based function ?
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
print("%f percent of twitter users feel positive about %s"%(pos,query))
print("%f percent of twitter users feel negative about %s"%(neg,query))
print("%f percent of twitter users feel neutral about %s"%(neu,query))

# Create a Wordcloud from the tweets
all_words = ' '.join([text for text in cleaned])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# Create list of all individual words
list_all_words = all_words.split()

# Remove words that are just numbers eg. years
list_all_words = [w for w in list_all_words if not w.isnumeric()]

# Find the 10 most common words in the tweets
common_words = Counter(list_all_words).most_common(10)
print('Top 10 most common words: ' + str(common_words))

# TODO: Add a function to see subjectivity in the tweets?
