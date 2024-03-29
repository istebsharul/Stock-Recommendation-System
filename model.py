from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tweepy
import math
import nltk
import numpy as np
from pandas import Series, DataFrame
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import timedelta
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import time

nltk.download('punkt')


def moving_avg(df):
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
    df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    df = df[['Close', 'HighLoad', 'Change', 'Volume']]

    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    df['rets'] = df['Close'] / df['Close'].shift(1) - 1

    return df

def make_predictions(df):

    df['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    df['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

    # Drop missing value
    df.fillna(value=-99999, inplace=True)
    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(df)))
    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    df['label'] = df[forecast_col].shift(-forecast_out)
    # X = np.array(df.drop(['label'], 1))
    X = np.array(df.drop('label', axis=1))
    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)
    # Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    # Separate label and identify it as y
    y = np.array(df['label'])
    y = y[:-forecast_out]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Linear regression
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_train)

    # KNN Regression
    model_knn = KNeighborsRegressor(n_neighbors=2)
    model_knn.fit(X_train, y_train)

    # Bayesian Ridge Regression
    model_by = BayesianRidge()
    model_by.fit(X_train, y_train)


    # LSTM regression
    model_lstm = Sequential()
    model_lstm.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model_lstm.add(Dense(units=1))
    model_lstm.compile(optimizer='adam', loss='mean_squared_error')

    # Reshape X_train and X_test for LSTM
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model_lstm.fit(X_train_lstm, y_train, epochs=10, batch_size=32)

    # Create confindence scores
    confidencereg = model.score(X_test, y_test)
    confidence_model_knn = model_knn.score(X_test, y_test)
    confidence_model_by = model_by.score(X_test, y_test)
    # LSTM requires a different approach to obtain confidence
    confidence_model_lstm = model_lstm.evaluate(X_test_lstm, y_test)

    reg = confidencereg * 100
    knn = confidence_model_knn * 100
    by = confidence_model_by * 100
    lstm = (1 - confidence_model_lstm) * 100  # LSTM returns loss, so use (1 - loss) as confidence

    score = "Regression {}\nKNN {}\nBayesian {}\nLSTM {}\n".format(reg, knn, by, lstm)
    # Create new columns
    forecast_reg = model.predict(X_forecast)
    forecast_knn = model_knn.predict(X_forecast)
    forecast_by = model_by.predict(X_forecast)

    # Reshape X_forecast for LSTM
    X_forecast_lstm = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))
    forecast_lstm = model_lstm.predict(X_forecast_lstm)

    # Process all new columns data
    df['Forecast_reg'] = np.nan

    last_date = df.iloc[-1].name
    last_unix = datetime.strptime(last_date, '%Y-%m-%d')
    next_unix = last_unix + timedelta(days=1)

    for i in forecast_reg:
        next_date = next_unix
        next_unix += timedelta(days=1)
        df.loc[next_date] = [np.nan for _ in range(len(df.columns))]
        df['Forecast_reg'].loc[next_date] = i

    df['Forecast_knn'] = np.nan

    last_date = df.iloc[-26].name
    last_unix = last_date
    next_unix = last_unix + timedelta(days=1)

    for i in forecast_knn:
        next_date = next_unix
        next_unix += timedelta(days=1)
        df['Forecast_knn'].loc[next_date] = i

    df['forecast_by'] = np.nan

    last_date = df.iloc[-26].name
    last_unix = last_date
    next_unix = last_unix + timedelta(days=1)

    for i in forecast_by:
        next_date = next_unix
        next_unix += timedelta(days=1)
        df['forecast_by'].loc[next_date] = i

    df['forecast_lstm'] = np.nan

    last_date = df.iloc[-26].name
    last_unix = last_date
    next_unix = last_unix + timedelta(days=1)

    for i in forecast_lstm:
        next_date = next_unix
        next_unix += timedelta(days=1)
        df['forecast_lstm'].loc[next_date] = i



    return df


def retrieving_tweets_polarity(symbol):
    # AAAAAAAAAAAAAAAAAAAAAOWtsgEAAAAAhikf7yKsz2eSVGkk2JWRk4M%2BJIc%3DyReSAYMSQOmiEaHDobQHoU74r1jwpQ47G7X7Asf2yrKsPDEnke
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_token_secret = ''

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    user = tweepy.API(auth, wait_on_rate_limit=True)

    # tweets = tweepy.Cursor(user.search, q=str(
    #     symbol), tweet_mode='extended', lang='en').items(100)

    tweets = tweepy.Cursor(user.search_tweets, q=str(
        symbol), tweet_mode='extended', lang='en').items(100)

    tweet_list = []
    print(tweet_list)
    global_polarity = 0
    for tweet in tweets:
        tw = tweet.full_text
        blob = TextBlob(tw)
        polarity = 0
        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity
            global_polarity += sentence.sentiment.polarity
        tweet_list.append(tw)

    global_polarity = global_polarity / len(tweet_list)
    return global_polarity
