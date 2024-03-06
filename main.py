import shutil
import urllib3
from datetime import datetime
import model
from datetime import timedelta
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from dash import html
from dash import dcc
import dash
from flask import Flask
import pandas as pd
import base64
import os
import sys


sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)))
os.chdir(os.path.realpath(os.path.dirname(__file__)))

# import dash_core_components as dcc
# import dash_html_components as html


UPLOAD_DIRECTORY = "../dash/data"

if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


external_stylesheets = [dbc.themes.BOOTSTRAP]

# Normally, Dash creates its own Flask server internally. By creating our own,
# we can create a route for downloading files directly:
server = Flask(__name__)
app = dash.Dash(server=server, external_stylesheets=external_stylesheets)

controls = dbc.Card(
    [
        dbc.CardBody(
            [
                dbc.Form(
                    [
                        dbc.CardGroup(  # Change from dbc.FormGroup to dbc.CardGroup
                            [
                                dbc.Label("Select Stock", style={
                                          "text-align": "center"}),

                                dcc.Dropdown(
                                    id="my-dropdown",
                                    options=[
                                        {'label': 'Google', 'value': 'GOOGL'},
                                        {'label': 'Coke', 'value': 'COKE'},
                                        {'label': 'Tesla', 'value': 'TSLA'},
                                        {'label': 'Apple', 'value': 'AAPL'},
                                        {'label': 'AMD', 'value': 'AMD'},
                                        {'label': 'Microsoft Corporation',
                                            'value': 'MSFT'},
                                        {'label': 'AT&T Inc.', 'value': 'T'}
                                    ],
                                    value='GOOGL',
                                    style={'border-radius': '5px', 'border': '1px solid #ced4da', 'padding': '0.375rem 0.75rem',
                                           'font-size': '1rem', 'line-height': '1.5', 'width': '100%'}  # Inline CSS
                                ),
                                html.Br(),
                            ]
                        ),
                    ]
                ),
            ]
        ),
    ],
    className="mb-3",
    style={'width': '50%', 'background-color': '#E5BAFF'}
)

app.layout = dbc.Container(
    [
        html.H1("Stock Recommendation System", style={'background-color': '#3B005F', 'color': '#FFFFFF',
                'textAlign': 'center', 'padding': '1%', 'margin': '0px'}),
        html.Hr(),

        dbc.Alert(
            [
                html.H4("Stock Predictions Using Machine Learning!",
                        className="alert-heading", style={"color": "white"}),
            ],
            color="#3B005F",
            style={"textAlign": "center",
                   "width": "max-content", "margin": "auto", "margin-bottom": "1%"},
        ),

        dbc.Row(
            [
                dbc.Row(controls, justify="center"),

                dbc.Row(dcc.Graph(id="my-graph")),
            ],
            justify="center",
        ),

        html.Div(
            id='my-div', style={'textAlign': 'center', 'color': 'red', 'font': '16px'}),

        html.Br(),

    ],
    fluid=True,
    style={"background-color": "#333131",
           "display": "flex", "flex-direction": "column", "justify-content": "center", "align-item": "center"}
)


@app.callback(Output('my-graph', 'figure'), [Input('my-dropdown', 'value')])
def get_data(selected_dropdown_value):

    date = datetime.today()
    ts = datetime.timestamp(date)
    start = int(ts)

    tss = datetime.today() - timedelta(days=3650)
    tss = datetime.timestamp(tss)
    end = int(tss)
    end

    url = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}&period2={}&interval=1d&events=history'.format(
        selected_dropdown_value, end, start)
    c = urllib3.PoolManager()
    filename = "../dash/data/{}.csv".format(selected_dropdown_value)

    with c.request('GET', url, preload_content=False) as res, open(filename, 'wb') as out_file:
        shutil.copyfileobj(res, out_file)

    data = pd.read_csv("../dash/data/{}.csv".format(selected_dropdown_value))

    # data = yf.download(selected_dropdown_value, start=datetime(2008, 5, 5), end=datetime.now())

    dff = pd.DataFrame(data)
    dfff = dff.set_index('Date')

    df = model.moving_avg(dfff)

    dff = model.make_predictions(dfff)

    return {
        'data': [{
            'x': df.index,
            'y': df['Close'],
            'name': 'Close'
        },
            {
            'x': df.index,
            'y': df['MA10'],
            'name': 'MA10'

        },
            {
            'x': df.index,
            'y': df['MA30'],
            'name': 'MA30'

        },
            {
            'x': df.index,
            'y': df['MA50'],
            'name': 'MA50'

        },
            {
            'x': df.index,
            'y': df['rets'],
            'name': 'Returns'

        },
            {
            'x': dff.index,
            'y': dff['Forecast_reg'],
            'name': 'Regression',

        },
            {
            'x': dff.index,
            'y': dff['Forecast_knn'],
            'name': 'KNN',


        },
            {
            'x': dff.index,
            'y': dff['forecast_by'],
            'name': 'Bayesian',


        }
        ],
        'layout': {'margin': {'l': 60, 'r': 60, 't': 30, 'b': 30}, 'title': 'Stock Data Visualization', 'align': 'center'}
    }


@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input('my-dropdown', 'value')]
)
def sentiment(input_value):

    polarity = model.retrieving_tweets_polarity(input_value)

    if polarity > 0:
        return 'According to the predictions and twitter sentiment analysis -> Investing in "{}" is a GREAT idea!'.format(str(input_value))

    elif polarity < 0:
        return 'According to the predictions and twitter sentiment analysis -> Investing in "{}" is a BAD idea!'.format(str(input_value))

    return 'According to the predictions and twitter sentiment analysis -> Investing in "{}" is a BAD idea!'.format(str(input_value))


if __name__ == "__main__":
    app.run_server(debug=True, port=8888)
