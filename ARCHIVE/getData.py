''''
Link to Github repository:

https://github.com/Daniilab/DSCI510_Final_Project.git

'''

from psaw import PushshiftAPI                               #Importing wrapper library for reddit(Pushshift)
import datetime as dt                                       #Importing library for date management
import pandas as pd                                         #Importing library for data manipulation in python
import matplotlib.pyplot as plt      
import yfinance as yf                    #Importing library for creating interactive visualizations in Python
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import statsmodels.formula.api as smf
import textwrap
import numpy as np
nltk.download('vader_lexicon') 
sia = SentimentIntensityAnalyzer()

api = PushshiftAPI()  #establishing the API 

#Creating a stock price data frame function

def Stock_Price_Data(ticker, after, before):
    data = yf.download(ticker,after,before)

    stock_price_df = pd.DataFrame(data)
    stock_price_df = stock_price_df.reset_index() #making Date a column rather than an index
    stock_price_df['Date'] = stock_price_df['Date'].astype(str) #convert Date to string format to prepare for merge
    
    return stock_price_df


def Reddit_Comment_Data2 (word_to_check, subbreddit, limit, before, after):

#grabbing desired comments using the Pushshift API
    comments = api.search_comments(q = word_to_check, subreddit= subreddit, limit= limit, before=before, after=after)
    comments_list=[]

#within the grabbed comments, I only want each comment's score, date posted, and text
    for comment in comments:
        comments_list.append(
            {"score": comment.score, "Date":comment.created, "comment_text": comment.body}
        )

    #turning the list into a master data frame
    comments_df=pd.DataFrame(comments_list)

    comments_df["Date"] = pd.to_datetime(comments_df["Date"], unit="s").dt.date

    # creating a new column with the compound sentiment score of each comment in the data frame
    comments_df['Sentiment'] = comments_df['comment_text'].apply(lambda x: sia.polarity_scores(x)['compound'])

    #multipling the compound sentiment score by the net upvote score to create a 'weighted sentiment' for each comment
    comments_df['Weighted Sentiment'] = comments_df['Sentiment']*comments_df['score']

    #creating sentiment df that is grouped by date
    grouped_weighted_sentiment_df = comments_df.groupby(by="Date")["Weighted Sentiment"].sum()
    pd.DataFrame(grouped_weighted_sentiment_df)

    #reseting the index and turning the data column into a string to prepare for a data frame merge
    grouped_weighted_sentiment_df = grouped_weighted_sentiment_df.reset_index()
    grouped_weighted_sentiment_df['Date'] = grouped_weighted_sentiment_df['Date'].astype(str)

    #creating score df that is grouped by date
    grouped_score_df = comments_df.groupby(by="Date")["score"].sum()
    pd.DataFrame(grouped_score_df)

   #reseting the index and turning the data column into a string to prepare for a data frame merge
    grouped_score_df = grouped_score_df.reset_index()
    grouped_score_df['Date'] = grouped_score_df['Date'].astype(str)

    #merging the two data frames
    score_sentiment_grouped_df = pd.merge(grouped_score_df, grouped_weighted_sentiment_df, on='Date')

    return score_sentiment_grouped_df

