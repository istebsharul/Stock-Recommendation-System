import tweepy
import math
import nltk
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime
from datetime import timedelta
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import LSTM,GRU, Dense, Dropout
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingRegressor
from tcn import TCN, tcn_full_summary

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


    # GRU regression
    model_gru = Sequential()
    model_gru.add(GRU(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
    model_gru.add(Dense(units=1))
    model_gru.compile(optimizer='adam', loss='mean_squared_error')

    # MLP regression
    model_mlp = Sequential()
    model_mlp.add(Dense(units=64, activation='relu',input_dim=X_train.shape[1]))
    model_mlp.add(Dropout(0.2))
    model_mlp.add(Dense(units=1))
    model_mlp.compile(optimizer='adam', loss='mean_squared_error')

    # Reshape X_train and X_test for LSTM
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model_lstm.fit(X_train_lstm, y_train, epochs=20, batch_size=32)


    # Reshape X_train and X_test for GRU
    X_train_gru = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_gru = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model_gru.fit(X_train_gru, y_train, epochs=20, batch_size=32)


    # TCN model
    model_tcn = Sequential([
        TCN(input_shape=(X_train.shape[1], 1), return_sequences=False),
        Dense(1)
    ])
    model_tcn.compile(optimizer='adam', loss='mean_squared_error')

    # Train TCN model
    model_tcn.fit(X_train, y_train, epochs=20, batch_size=32)

    # Train MLP model
    model_mlp.fit(X_train, y_train, epochs=50, batch_size=32)


    # GBM Regression
    model_gbm = GradientBoostingRegressor()
    model_gbm.fit(X_train, y_train)


    # Create confindence scores
    confidencereg = model.score(X_test, y_test)
    confidence_model_knn = model_knn.score(X_test, y_test)
    confidence_model_by = model_by.score(X_test, y_test)

    
    # LSTM requires a different approach to obtain confidence
    confidence_model_lstm = model_lstm.evaluate(X_test_lstm, y_test)

    # GRU requires a different approach to obtain confidence
    confidence_model_gru = model_gru.evaluate(X_test_gru, y_test)

    # Evaluate TCN model on test data
    confidence_tcn = model_tcn.evaluate(X_test, y_test)

    # Evaluate MLP model on test data

    confidence_model_mlp = model_mlp.evaluate(X_test, y_test)

    # Calculate confidence score
    confidence_model_gbm = model_gbm.score(X_test, y_test)


    reg = confidencereg * 100
    knn = confidence_model_knn * 100
    by = confidence_model_by * 100
    lstm = (1 - confidence_model_lstm) * 100  # LSTM returns loss, so use (1 - loss) as confidence
    gru = (1 - confidence_model_gru) * 100
    tcn = confidence_tcn * 100
    mlp = (1 - confidence_model_mlp) * 100
    gbm = confidence_model_gbm * 100

    score = "Regression {}\nKNN {}\nBayesian {}\nLSTM {}\nGRU {}\nMLP {}\nTCN {}\nGBM {}\n".format(reg, knn, by, lstm,gru,mlp,tcn,gbm)
    # Create new columns
    forecast_reg = model.predict(X_forecast)
    forecast_knn = model_knn.predict(X_forecast)
    forecast_by = model_by.predict(X_forecast)

    # Reshape X_forecast for LSTM
    X_forecast_lstm = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))
    forecast_lstm = model_lstm.predict(X_forecast_lstm)


    # Reshape X_forecast for GRU
    X_forecast_gru = np.reshape(X_forecast, (X_forecast.shape[0], X_forecast.shape[1], 1))
    forecast_gru = model_gru.predict(X_forecast_gru)

    # Reshape X_forecast for MLP
    forecast_mlp = model_mlp.predict(X_forecast)

    # Make forecasts with TCN
    forecast_tcn = model_tcn.predict(X_forecast)

    # Reshape X_forecast for GBM
    X_forecast_gbm = X_forecast  # No need to reshape for GBM



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

    # Process forecasts
    df['Forecast_tcn'] = np.nan
    last_date = df.iloc[-26].name
    next_unix = last_date + timedelta(days=1)

    for i in forecast_tcn:
        next_date = next_unix
        next_unix += timedelta(days=1)
        # df.loc[next_date] = [np.nan for _ in range(len(df.columns))]
        df['Forecast_tcn'].loc[next_date] = i


    # Process forecasts
    df['Forecast_mlp'] = np.nan
    last_date = df.iloc[-26].name
    next_unix = last_date + timedelta(days=1)

    for i in forecast_mlp:
        next_date = next_unix
        next_unix += timedelta(days=1)
        # df.loc[next_date] = [np.nan for _ in range(len(df.columns))]
        df['Forecast_mlp'].loc[next_date] = i

        
    # Create new column
    df['Forecast_gbm'] = np.nan

    # Make forecasts
    forecast_gbm = model_gbm.predict(X_forecast_gbm)

    # Process forecasts
    last_date = df.iloc[-26].name
    next_unix = last_date + timedelta(days=1)

    for i in forecast_gbm:
        next_date = next_unix
        next_unix += timedelta(days=1)
        # df.loc[next_date] = [np.nan for _ in range(len(df.columns))]
        df['Forecast_gbm'].loc[next_date] = i

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

    df['forecast_gru'] = np.nan

    last_date = df.iloc[-26].name
    last_unix = last_date
    next_unix = last_unix + timedelta(days=1)

    for i in forecast_gru:
        next_date = next_unix
        next_unix += timedelta(days=1)
        df['forecast_gru'].loc[next_date] = i

        

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
