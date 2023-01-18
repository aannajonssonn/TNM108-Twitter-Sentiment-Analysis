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
from textblob import Word
from wordcloud import WordCloud             # Create a word cloud to visualize the most common words
from better_profanity import profanity      # Remove profanity from tweets
from collections import Counter             # Count the number of times a value appears in a list

# Importing libraries for the GUI
import PySimpleGUI as sg                    # To make the GUI  

# Define the stop words
stop_words = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're','s',
             'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves', 'rt', 'lol', 'yet',
             'see', 'get', 'go', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
             'eight', 'nine', 'ten', 'also', 'would', 'could', 'should', 'may', 'might',
             'must', 'shall', 'will', 'us', 'im', 'ive', 'got', 'set']

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

# Continue with the Twitter Sentiment Analysis, pre-processing

# Remove retweets
search_term = query + " -filter:retweets"

# Get the latest 100 tweets on the topic
# HÄR KAN VI ÄNDRA SPRÅK, ANTAL TWEETS, OSV
tweets = tweepy.Cursor(api.search_tweets, q = search_term, lang = "en").items(500)

# Create a list of tweets, the users, and their location
list1 = [[tweet.text, tweet.user.screen_name, tweet.user.location] for tweet in tweets]

# Convert the list into a dataframe
df = pd.DataFrame(data=list1, 
                    columns=['tweets','user', "location"])

# Save the uncleaned tweets to a csv file
df['tweets'].to_csv('dataset/uncleaned_tweets.csv', index=False, encoding = 'utf-8')

# Preprocessthe data by creating a function to clean the tweets. Remove profanity, unnecessary characters, spaces, and stopwords.
def clean_tweets(df, stop_words):
    df['tweets'] = df['tweets'].apply(lambda x: ' '.join(x.lower() for x in x.split()))
    # Remove profanity
    df['tweets'] = df['tweets'].apply(lambda x: profanity.censor(x))
    # Remove links
    df['tweets'] = df['tweets'].str.replace(r'http\S+', '') 
    # Removing stop words
    df['tweets'] = df['tweets'].apply(lambda x: ' '.join(x for x in x.split() if x not in stop_words))
    # Lemmatization
    df['tweets'] = df['tweets'].apply(lambda x: ' '.join([Word(x).lemmatize() for x in x.split()]))
    return df

# Run the list of tweets through the clean_tweet function and display the tweets
cleaned = clean_tweets(df, stop_words)
#cleaned

# Save the cleaned tweets to a csv file
cleaned['tweets'].to_csv('dataset/cleaned_tweets.csv', index=False, encoding = 'utf-8')

# Convert only the tweets to a list
cleaned_li = cleaned['tweets'].tolist()

# Define the sentiment objects using TextBlob
sentiment_objects_tb = [TextBlob(tweet) for tweet in cleaned_li]
sentiment_objects_tb[0].polarity, sentiment_objects_tb[0]

# Create a list of polarity values and tweet text
sentiment_values_tb = [[tweet.sentiment.polarity, str(tweet)] for tweet in sentiment_objects_tb]

# Print the value of the 0th row.
sentiment_values_tb[0]

# Print all the sentiment values
sentiment_values_tb[0:99]

# Create a dataframe of each tweet against its polarity
sentiment_df = pd.DataFrame(sentiment_values_tb, columns=["polarity", "tweet"])
sentiment_df

# Save the polarity column as 'n'.
n = sentiment_df["polarity"]

# Convert this column into a series, 'm'. 
m = pd.Series(n)
m

# Initialize variables, 'pos', 'neg', 'neu'.
pos = 0
neg = 0
neu = 0

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
        
# Divide the number of each by the total number of tweets to get the percentage of each.
pos = (pos/len(m))*100
neg = (neg/len(m))*100
neu = (neu/len(m))*100

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
all_words = ' '.join([text for text in cleaned_li])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, collocations=False).generate(all_words)
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
