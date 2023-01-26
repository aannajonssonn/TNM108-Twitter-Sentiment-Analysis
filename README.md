# TNM108 Project - Twitter Sentiment Analysis
This is a project made for the university course TNM108 - Machine Learning for Social Media at Linköpings University 2022. 

The project is made by Anna Jonsson and Amanda Bigelius, and the goal is to make a Twitter Sentiment Analysis Algorithm.

In the end, the project resulted in two different solutions. One solution where TextBlob, a lexicon-based method, was used, and one where Logistic Regression was used.

## Twitter Sentiment Analysis using TextBlob
The algorithm will be heavily based on [Nikita Silaparasetty's](https://github.com/nikitasilaparasetty) code from [this tutorial](https://medium.com/@nikitasilaparasetty/twitter-sentiment-analysis-for-data-science-using-python-in-2022-6d5e43f6fa6e)

Her repository for the tutorial can be found [here](https://github.com/nikitasilaparasetty/Twitter-Sentiment-Analysis-Projects-2022-)

### Our modifications and thoughts
Our first modification was to move all the API_KEYS to a separate file in order to be able to uplead the code on GitHub. 

We also added our own list of stopwords since the NLTK stopwords removed some words we found important for the classification. 

We added a way to check the most frequent words from the tweets, without the query and only using words longer than 2 characters. 
Later on we added filtered out the NLTK stopwords on our most common words, since the analysis was done and these stopwords weren't relevant when looking at the word frequency. 
Then we displayed it as a bar plot.

Lastly we added a simple GUI to make it more intuitive for the user where to put the query. 

Our assignment was to make a algorithm using machine learning, and although TextBlob is a good tool, it doesn't cover our needs for this assignment. 

#### Graphical User Interface
The GUI has been made with the library PySimpleGUI, and this [stackoverflow answer](https://stackoverflow.com/a/66537814) was very helpful.

### Requirements :hammer_and_wrench:
In order for this algorithm to work you need to have python installed on your computer, as well as the following libraries:
- [Tweepy](https://www.tweepy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/index.html)
- [WordCloud](http://amueller.github.io/word_cloud/)
- [Better_Profanity](https://github.com/snguyenthanh/better_profanity)
- [PySimpleGUI](https://www.pysimplegui.org/en/latest/)
- [NLTK](https://www.nltk.org/index.html)
- [Collection](https://docs.python.org/3/library/collections.html#)

#### Install libraries using pip
To install the libraries using pip, write the following command lines one by one:
- Tweepy: ```pip install tweepy```
- Matplotlib: ```pip install matplotlib```
- Pandas: ```pip install pandas```
- TextBlob: ``` pip install -U textblob ``` as well as ```python -m textblob.download_corpora``` to download the necessary NLTK corpora.
- WordCloud: ```pip install wordcloud```
- Better Profanity: ```pip install better_profanity```
- PySimpleGUI: ```pip install pysimplegui```
- NLTK: ```pip install nltk```
- Collection: ```pip install collection```


## Twitter Sentiment Analysis using Logistic Regression
The algorithm will be heavily based on [Kate Arbuzova's](https://medium.com/@kate.arbuzova) code from [this tutorial](https://www.kaggle.com/code/katearb/sentiment-analysis-in-twitter-93-test-acc/notebook).

The dataset used for this method can be found [on Kaggle](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

### Our modifications and thoughts
Our first modifications to Kate's code was to only look at the Logistic Regression methods she used. 

We also increased the numbers of features to 10,000 - this was probably a bad move, but we still did it.

Then we commented out a lot of code, just to make the program print less stuff. 

The runtime for this was extremely long, so we would recommend scaling everything down. 

### Requirements :hammer_and_wrench:
In order for this algorithm to work you need to have python installed on your computer, as well as the following libraries:
- [Scikit-learn](https://scikit-learn.org/stable/index.html)
- [SciPy](https://scipy.org/)
- [NLTK](https://www.nltk.org/index.html)
- [Statsmodels](https://www.statsmodels.org/stable/index.html)
- [Emoji](https://pypi.org/project/emoji/)
- [Regex](https://docs.python.org/3/library/re.html)
- [Spacy](https://spacy.io/models/en)
- [TQDM](https://tqdm.github.io/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Pickle](https://docs.python.org/3/library/pickle.html)
- [Seaborn](https://seaborn.pydata.org/)

#### Install libraries using pip
To install the libraries using pip, write the following command lines one by one:
- Scikit-learn: ```pip install scikit-learn```
- SciPy: ```pip install scipy```
- NLTK: ```pip install nltk```
- Statsmodels: ```pip install statsmodels```
- Emoji: ```pip install emoji```
- Regex: ```pip install regex```
- Spacy: ```pip install spacy```
- TQDM: ```pip install tqdm```
- Matplotlib: ```pip install matplotlib```
- Pandas: ```pip install panda```
- Pickle: ```pip install pickle```
- Seaborn: ```pip install seaborn```

