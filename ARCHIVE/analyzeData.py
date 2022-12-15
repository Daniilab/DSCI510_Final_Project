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


#regression analysis
def createModels(mergedData2):
    # creating two models that have score and weighted sentiment as two different predictors of closing price
    model1 = smf.ols('Close ~ score', mergedData2)
    model2 = smf.ols('Close ~ Weighted_Sentiment', mergedData2)

    results1 = model1.fit()
    results2 = model2.fit()

    # display the results
    print(results1.summary())
    print(results2.summary())

    return model1, model2



#regression visualizations

def regressionVisualizationWS(mergedData2):
    # create a scatter plot of the data
    mergedData2.plot(kind='scatter', x='Weighted_Sentiment', y='Close')

    # fitting the linear regression model to the data
    slope, intercept = np.polyfit(mergedData2['Weighted_Sentiment'], mergedData2['Close'], 1)

    # adding a regression line to the scatter plot
    plt.plot(mergedData2['Weighted_Sentiment'], slope*mergedData2['Weighted_Sentiment'] + intercept, color='red')

    #adding a title
    plt.title("Closing Price vs Daily Summed Weighted Sentiment")


    # add a description to the graph
    description = textwrap.wrap("This figure shows a scatter plot of the closing price of Game Stop vs weighted summed sentiment score of that day, and a line of best fit used for the linear regression. This graph demonstrates how positive sentiment towards Game Stop on Reddit correlates to a price increase of its stock.", width=60)

    plt.text(-20, -45,'\n'.join(description), fontsize=10, ha='left', va='bottom')

    #saving the sigure as a new file
    plt.savefig('regression_plot1.jpeg',dpi=300, bbox_inches = "tight")
    # plt.savefig('regression_plot1.jpeg')

    # show the plot
    plt.show()



def regressionVisualizationScore(mergedData2):
    # creating a scatter plot of the data
    mergedData2.plot(kind='scatter', x='score', y='Close')

    # fitting the linear regression model to the data
    slope, intercept = np.polyfit(mergedData2['score'], mergedData2['Close'], 1)

    # adding a regression line to the scatter plot
    plt.plot(mergedData2['score'], slope*mergedData2['score'] + intercept, color='red')

    #creating a title
    plt.title('Closing Price vs Daily Summed Score')

    # add a description to the graph
    description = textwrap.wrap("This figure shows a scatter plot of the closing price of Game Stop vs summed score (net upvotes) of that day, and a line of best fit used for the linear regression. This graph utilizes an alternative variable to demonstrate how positive sentiment towards Game Stop on Reddit correlates to a price increase of its stock.", width=60)

    plt.text(-20, -45,'\n'.join(description), fontsize=10, ha='left', va='bottom')

    #saving figure as new file
    plt.savefig('regression_plot2.jpeg',dpi=300, bbox_inches = "tight")

    # show the plot
    plt.show()


#time series visualizations

def timeSeriesVisualizationScore(mergedData2):
    fig, ax = plt.subplots() # Create the figure and axes object

    # Plotting date on the x axis and percent change of score (net upvotes) on the y axis
    mergedData2.plot(x = 'Date', y = 'Pct Change of Score', ax = ax) 

    #adding the percent change of closing price to the graph
    mergedData2.plot(x = 'Date', y = 'Pct Change of Closing Price', ax = ax, secondary_y = True) 
    plt.title('Percent Change of Score and Closing Price vs Time')

    # add a description to the graph
    description = textwrap.wrap("This figure shows how the summed daily score of comments mentioning Game Stop and Game Stop's stock price varies with time. This graph shows that changes in price preceed changes in score, suggesting that stock price changes is the leading variable.", width=60)

    plt.text(0, -280,'\n'.join(description), fontsize=10, ha='left', va='bottom')

    #saving the figure as a new file
    plt.savefig('time_series_plot1.jpeg',dpi=300, bbox_inches = "tight")


def timeSeriesVisualizationWS(mergedData2):
    fig, ax = plt.subplots() # Create the figure and axes object

    # Plotting date on the x axis and percent change of weighted sentiment on the y axis
    mergedData2.plot(x = 'Date', y = 'Pct Change of WS', ax = ax) 

    #adding the percent change of closing price to the graph
    mergedData2.plot(x = 'Date', y = 'Pct Change of Closing Price', ax = ax, secondary_y = True) 
    plt.title('Percent Change of WS and Closing Price vs Time')

    # adding a description to the graph
    description = textwrap.wrap("This figure shows how the weighted daily sum of sentiment scores of comments mentioning Game Stop and Game Stop's stock price varies with time. This graph displays less of a clear relationship between the two variables, and suggests that the sentiment analysis package used may not be accurate at determining the true sentiment of comments regarding Game Stop.", width=60)

    plt.text(0, -340,'\n'.join(description), fontsize=10, ha='left', va='bottom')


    plt.savefig('time_series_plot2.jpeg',dpi=300, bbox_inches = "tight")