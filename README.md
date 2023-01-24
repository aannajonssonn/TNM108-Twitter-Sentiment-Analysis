# TNM108 Project - Twitter Sentiment Analysis
This is a project made for the university course TNM108 - Machine Learning for Social Media at Linköpings University 2022. 

The project is made by Anna Jonsson and Amanda Bigelius, and the goal is to make a Twitter Sentiment Analysis Algorithm.

If you're interested in the process of this project, we have a little blog here: https://codelikealady.blogspot.com/ 

## Inspiration and sources :brain:
### Twitter Sentiment Analysis using TextBlob
The algorithm will be heavily based on [Nikita Silaparasetty's](https://github.com/nikitasilaparasetty) code from [this tutorial](https://medium.com/@nikitasilaparasetty/twitter-sentiment-analysis-for-data-science-using-python-in-2022-6d5e43f6fa6e)

Her repository for the tutorial can be found [here](https://github.com/nikitasilaparasetty/Twitter-Sentiment-Analysis-Projects-2022-)

### Twitter Sentiment Analysis using ...
This algorithm uses the tweets scraped during the previous step, as well as tweets from [this dataset](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis).

### Graphical User Interface
The GUI has been made with the library PySimpleGUI, and this [stackoverflow answer](https://stackoverflow.com/a/66537814) was very helpful.

## Requirements :hammer_and_wrench:
In order for this algorithm to work you need to have python installed on your computer, as well as the following libraries:
- [NumPy](https://numpy.org/)
- [Tweepy](https://www.tweepy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [TextBlob](https://textblob.readthedocs.io/en/dev/index.html)
- [WordCloud](http://amueller.github.io/word_cloud/)
- [Better_Profanity](https://github.com/snguyenthanh/better_profanity)
- [PySimpleGUI](https://www.pysimplegui.org/en/latest/)
- [NLTK](https://www.nltk.org/index.html)
- [Flair](https://github.com/flairNLP/flair)
- [Collection](https://docs.python.org/3/library/collections.html#)

### Install libraries using pip
To install the libraries using pip, write the following command lines one by one:

- Numpy: ```pip install numpy```
- Tweepy: ```pip install tweepy```
- Matplotlib: ```pip install matplotlib```
- Pandas: ```pip install pandas```
- TextBlob: ``` pip install -U textblob ``` as well as ```python -m textblob.download_corpora``` to download the necessary NLTK corpora.
- WordCloud: ```pip install wordcloud```
- Better Profanity: ```pip install better_profanity```
- PySimpleGUI: ```pip install pysimplegui```
- NLTK: ```pip install nltk```
- Flair: ```pip install flair```
- Collection: ```pip install collection```