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


#merging the reddit data frame with the stock price data frame
def mergeData(r2, stockpriceGme):
    mergedData2 = pd.merge(r2, stockpriceGme, on='Date')

    #creating percent change columns to prepare for time series visualizations in the analysis section
    mergedData2['Pct Change of Score'] = mergedData2['score'].pct_change()*100
    mergedData2['Pct Change of Closing Price'] = mergedData2['Close'].pct_change()*100
    mergedData2.rename(columns={'Weighted Sentiment': 'Weighted_Sentiment'}, inplace=True)
    mergedData2['Pct Change of WS'] = mergedData2['Weighted_Sentiment'].pct_change()*100

    #deleting columns that I won't be using in analysis
    del mergedData2["Adj Close"]
    del mergedData2["Open"]
    del mergedData2["High"]
    del mergedData2["Low"]

    return mergedData2