from getData import Stock_Price_Data, Reddit_Comment_Data2
from mergeData import mergeData
from analyzeData import createModels, regressionVisualizationWS, regressionVisualizationScore, timeSeriesVisualizationScore,  timeSeriesVisualizationWS

# Import Libraries
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


if __name__ == '__main__':

    api = PushshiftAPI()  #establishing the API 

    # Get Data
    before = int(dt.datetime(2021,2,26,0,0).timestamp()) #.timestamp() converts the date to epoch time 
    after = int(dt.datetime(2021,1,1,0,0).timestamp())

    subreddit="wallstreetbets"
    limit=25000

    stockpriceGme = Stock_Price_Data("GME", after, before)
    r2 = Reddit_Comment_Data2("gme to the moon", subreddit, limit, before, after)

    # Merge Data
    mergeData2 = mergeData(r2, stockpriceGme)
    
    # Analyze Data
    model1, model2 = createModels(mergeData2)
    regressionVisualizationWS(mergeData2)
    regressionVisualizationScore(mergeData2)
    timeSeriesVisualizationScore(mergeData2)
    timeSeriesVisualizationWS(mergeData2)




