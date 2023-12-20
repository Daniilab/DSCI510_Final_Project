# Reddit’s Impact on Stock Prices

##  Research question: How does the number of Reddit posts on “r/wallstreetbets” regarding GameStop’s stock impact its stock price, if at all?

**Important Note: The Pushshift API stopped working towards the end of this project. Consequently, I was unable to run the code and grab the outputs and had to make do with what I already had. I included outputs from HW 4 which is fully up to date with my final project, as well as some screenshots. Please let me know if you need further clarification. **


Github repository: https://github.com/Daniilab/DSCI510_Final_Project.git 



Description: During the beginning of 2021, GameStop saw an incredible spike in its stock price, and many accredited this spike to the activity on the subreddit ‘r/wallstreetbets.’ There was an explosion of posts encouraging others to buy the stock in order to inflate its stock price. I would like to examine if the hypothesis that Reddit activity is correlated to stock prices is true. To do this, I will examine the daily Reddit activity regarding GameStop on r/wallstreetbets from January to February 2021 and see if there is a correlation to GameStop’s stock price during the same time frame via a linear regression analysis. 


# Dependencies

Using the anaconda navigator make sure you have these requirements installed:

psaw==0.1.0
datetime==2.8.2
pandas==1.4.2                                  
matplotlib.pyplot==3.5.1
yfinance==0.1.85                
nltk==3.7
statsmodels==0.13.2
textwrap==3.4
numpy==1.21.5


# Installation 

How to install the requirements necessary to run your project:  

Simply download the FINAL_PROJECT.ipynb file and make sure all libraries are installed

# Running the project

Download the Jupyter Notebook file and click "Run All." Note: at the time of me writing this, the Pushshift API is not connecting so the code will not run until it does connect. 

# Methodology

I collected data from two different sources: I collected data from Reddit using the Pushshift API and Stock Price Data from the Yahoo Finance API. 
Pushshict API: https://github.com/pushshift/api
Yahoo Finance API accessed from the yfinance package on Python
My goal was to collect data that would allow me to correlate the frequency of positive sentiment interactions towards GameStop expressed on Reddit to GameStop’s daily stock price
I decided to use “closing price” as the price variable. The reason for this is because my hypothesis is that closing price is impacted by the accumulation of interactions on Reddit for that day. I would need to get data on the price at the end of the day to be consistent with this hypothesis. 
Creating the Reddit Data Set
To acquire my Reddit data, I web scraped 25,000 comments from Reddit that were posted on the subreddit “r/wallstreetbets” from the time range of January 1, 2021 to February 26, 2021 containing words from the phrase “GME to the moon.” This was a phrase used countlessly on the subreddit which essentially translates to “let’s buy more GameStop so that we can inflate its stock price.” 
Note: I’m using the phrase “GME to the moon” instead of just “GME” because the former phrase is more specific and thus allows me to grab data over the course of several days. I will explain this in further detail in the challenges I encountered section. 
Note: I ended up with data spanning from January 14, 2021 to February 26, 2021 because the program searches for the 25,000 most recent comments, and stops once it reaches that limit.  
I created a large data frame of these comments which contained the date the comment was posted, and the score associated with the comment. On Reddit, a comment’s score is the difference between its total upvotes and downvotes, or the net number of upvotes. The score, therefore, represents the number of positive interactions associated with a comment. Thus, I felt that score would be an appropriate variable in my regression as each point on a score represents an individual who is contributing to the positive sentiment of Gamestop’s stock. 
From there, I made a new data frame that summed the scores of all comments for each day. Thus, on my new data frame, each row represents the total positive interactions associated with GameStop for that day. 
To make my project more advanced, I used the nltk library to compute the compound sentiment score of each comment and multiplied the sentiment score of each comment by the net upvote score (to give it a 'weight'), and added up those products on a daily basis. This allowed me to create the additional column titled “weighted sentiment" which represents the summed weighted daily sentiment score. 
Note: in this paper, the term “score” will strictly refer to the net upvotes on a comment, while “sentiment score” will strictly refer to the compound sentiment score as calculated by the nltk library.
Finally, I created percent change columns for score, weighted sentiment, and closing price which will be used when making a time series visualization.
Creating the Stock Price Data Set and Merging the Two Data Sets
To acquire my stock price data, I used the yfinance library to retrieve data of GME’s closing price per day during the same time frame (January and February 2021). 
To prepare for the merge I had to reindex the date columns of both data sets and convert them both to strings; then I merged the two data frames into one data set. This concludes my explanation of how I gathered and created my data set. 
How My Project Changed and Challenges I Encountered
One challenge that I encountered while trying to create a data set was dealing with the sheer amount of mentions of the stock “GME.” I needed to collect enough data such that I was grabbing comment data over the course of several days. However, because GME was wildly popular during the time frame that I’m webscrapping, I found that even if I webscrape 25,000 comments (which takes 6-7 minutes to run), I would still only acquire data for one day (meaning that there were at least 25 thousand comments about GME in one day. I realized that I needed to use a more specific phrase that was said less frequently so that if I webscrapped 25,000 comments I would have data over several days. I resolved this issue by using the phrase “gme to the moon” instead because this phrase was said less frequently, and thus allowed me to grab comment data over the course of several days. Furthermore, I believe all comments containing the words “gme to the moon” will inherently have a positive sentiment. This means that score will be a solid indicator of positive sentiment. 
Another challenge I encountered was when I attempted to merge the Reddit data set with the Stock Price data set. After creating two data sets with a date column in each, Python was not allowing me to merge them on the “Date” column. I eventually discovered that they were not merging because the Date columns were treated as indexes rather than true columns. I resolved this issue by resetting the index of each data frame such that the Date column became a real column rather than an idex. From there, I converted these columns into strings so that they could be of the same data type, which allowed me to perform the merge. 


# Visualization and Analysis

What Analysis I did:
I created two regression models: (1) Score (x) and Closing Price (y), and (2) Weighted Sentiment (x) and Closing Price (y). Each of these models has an OLS Regression output and a scatter plot with a line of best fit. 
To help infer a casual relationship, I also created two time series graphs: one with percent change of score & percent change of closing price and one with percent change of weighted sentiment & percent change of closing price. The purpose of this was to help visually identify if score/weighted sentiment was a leading or lagging variable to work towards finding a casual relationship. However, this is not a true statistical analysis and was used more for general curiosity. 
Observations:
Based on the p values, both models were significant, especially model 1. These models affirm the hypothesis that Reddit sentiment towards a stock is correlated with the stock’s price.
One major observation is that the “score” is a much stronger variable to use than the “weighted sentiment” variable. This is evident it that the model that uses “score” has a lower p value (at a 0.000 significance level) and is thus more significant, and has a greate R^2 value (at 0.436), which means that the variation of the x values better explains the variation of y values in the score model. Additionally, on the time series plot, the score data points follow the price data points much more closely than weighted sentiment scores follow the price; in the latter there is no clear relationship between the two but in the former there is a clear relationship. 
I believe that using score is a better indicator than using weighted sentiment because I believe the nltk library did a poor job of accurately assessing the sentiment. The subreddit r/wallstreetbets is full of sarcasm and humor, which I believe was difficult for the sentiment analyzer to decode because sarcastic and humorous statements often appear to be negative at face value but are meant to express positive sentiment. This suspicion of inaccuracy was reinforced when I looked at the actual text of the comments and the calculated sentiment score; there were several instances where I interpreted several comments to be positive while the analyzer calculated a neutral/negative score. Additionally, I believe that score (net upvotes) is a respectable variable because I believe that most, if not all, comments containing the words “GME to the moon” will be inherently positive.
Visually, it appears that score was a lagging indicator of closing price because the scores spiked/dipped one day after the closing price of the stock spiked/dipped in several regions of the graph. This implies that increased Reddit activity did not cause the price spike but was rather a symptom of the price spike. Nevertheless, more analysis is necessary to know for certain and no definitive conclusions can be drawn. 


# Conclusions Impact of findings: 

From my analysis, I discovered that there is a strong correlation between reddit activity regarding GameStop and GameStop’s stock price.
This is impactful because it suggests that social media can impact the stock market and therefore has an impact on our economy. Many Americans have their wealth invested in the stock market, and being able to make smart decisions with regards to investing is fundamental to people’s future health. From behavioral finance, we know that a company’s fundamentals is not the only factor that moves stock its price–there is a herd behavioral aspect that can also impact its stock price. This herd behavior can be hard to predict, and as a consequence, many people could lose money. My research can provide insight into how to predict stock price based on irrational market exuberance, which can be captured on social media. 
Challenges Encountered:
The analysis/visualization aspect of my project was more straightforward than the data gathering portion, so I ran into less problems. The biggest challenge I ran into was figuring out how to format the description text on my plots. The text was in an incorrectly placed on my graph and was not wrapping. I eventually was able to format it correctly through tinkering with the code and using trial and error until it was formatted correctly. 


# Future Work

Given more time, I would make my project more advanced by addressing its limitations. Here are some of the limitations and potential improvements I’ve identified:

Limitation #1: The model only found the relationship between Reddit activity and stock price for GameStop. We don’t know if the relationship holds true for other stocks.

Potential Improvement: Create a regression model that analyzes multiple stocks at once:
Incorporating a model with several stocks will allow me to discover more generally if activity on Reddit has a relationship to stocks prices.
The way that I would do this is by creating a dummy variable for each unique stock
The data table would be set up in a way that in addition to columns such as “Date,” “Closing Price” and “Score,” there would be columns titled with the stock name such “GME,” “AMC,” “TSLA.” Under each row in these columns, there will be either a 1 or a 0. When there is a 1, that means that the Closing price help in that row corresponds to that particular stock. When there is a 0, it means that the closing price in that row does not correspond to that stock. 
I would run a multiple regression analysis. If the coefficient of the score variable is still significant, it means that regardless of the stock, Reddit activity does have relationship with stock price. If the model is no longer significant that means there is not an overall independent effect. 
 
Limitation #2: While I found that a relationship exists, we don’t know for certain what caused what: did the stock price increase cause an increase in reddit activity around the stock or vice versa?

Potential Improvement: Creating a more robust time series analysis:
Based on my time series graph it appears that the leading indicator was the stock price. If I were to continue on with this project, I would incorporate multiple stocks into the analysis and look over a longer period of time. Doing so would allow me to have a better understanding of the causal relationship. 
I could also look into using the statistical methods such as path analysis, instrumental variable analysis, and time-series cross-sectional analysis. In the latter, I would combine time-series data with cross-sectional data and use a regression analysis to examine the relationship between Reddit activity and stock prices. This could potentially provide more detailed insights into the causal relationships between the variables and would allow me to control for potential confounders.
 
Limitation #3: Using the key phrase “to the moon,” while useful for the purpose of this project, has two flaws. The first is that I can’t use this phrase if I were to create a more advanced model that incorporates multiple stocks at once, because that phrase is specific to GameStop. The second problem is that this phrase is a derivative of Game Stop’s prior success and is thus is not an ideal independent variable. 
 
Potential Improvement: Use just the stock code name “GME” (or any other stock) instead of the phrase “gme to the moon.”
I would need to webscrape around 750,000 comments instead of 25,000 comments to account for the greater daily mentions of “GME” than “gme to the moon.” This would require alot more computing power but would ultimately return better data.
 
Limitation #4: The sentiment analysis package that I used appeared to not be incredibly accurate. As explained earlier, I believe that the nltk library does a poor job of understanding sarcasm and humor, resulting in inaccuracies.  
 
Potential Improvement: I would look into alternate ways to analyze sentiment:
Using an alternate pre-trained sentiment analysis model: such as TextBlob and VADER and see if there is a stronger correlation between the variables
Building a custom sentiment analysis model using machine learning techniques in Python. This would involve splitting the data into training and test sets, training a machine learning model on the training data, and evaluating the model on the test data.
Combining multiple methods: I could use one of Python’s pre-trained models to classify the overall sentiment of a text, and then use additional methods such as subjectivity analysis  or keyword analysis or to provide more detailed insights.
