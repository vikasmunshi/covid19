#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
from io import StringIO
from os import path
from sys import argv

import cufflinks as cf
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import requests
import requests_cache

from .data_corrections import data_corrections
from .urls import urls

# using simple time-based caching to prevent abuse of data providers
requests_cache.install_cache(path.split(path.split(__file__)[0])[1], expire_after=12 * 3600)


# retrieve data and return Pandas DataFrame
def data_frame_from_url(url_name: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(requests.get(urls[url_name]).content.decode()))


# Population Data
population = data_frame_from_url('population').set_index('Country').iloc[:, [-1]].dropna(0)
population.columns = ['Population']
population.Population = population.Population.astype(int)
for country, name_change, missing_data in data_corrections:
    if name_change:
        population.loc[name_change] = population.loc[country]
    if missing_data:
        population.loc[country] = missing_data
population = population / 10 ** 6
population.loc['Rest'] = population.loc['World'] - population.loc['China']

# Covid-19 Data
data_frames = {}
for name, data_frame in ((k, data_frame_from_url(k)) for k in urls.keys() if k != 'population'):
    data_frame = data_frame.drop(0)[['Country/Region', 'Province/State', 'Date', 'Value']]
    data_frame.Date = pd.to_datetime(data_frame.Date)
    data_frame.Value = data_frame.Value.astype(int)
    data_frame = data_frame.groupby(['Country/Region', 'Date']).sum().unstack()
    data_frame = pd.concat([data_frame, population], axis=1, join='inner').drop('Population', axis=1)
    data_frame = data_frame.transpose()
    data_frame.index = pd.MultiIndex.from_tuples(data_frame.index)
    data_frame = data_frame.rename_axis(index=['Value', 'Date']).reset_index(['Value']).drop('Value', axis=1)
    data_frame['World'] = data_frame.sum(axis=1)
    data_frame['Rest'] = data_frame['World'] - data_frame['China']
    data_frames[name] = data_frame
    data_frames[name + ' last 7 days'] = data_frames[name].diff(7).dropna()
    data_frames[name + ' per million'] = data_frame.div(population.Population, axis=1).dropna(axis=1)
    data_frames[name + ' per million last 7 days'] = data_frames[name + ' per million'].diff(7).dropna()
data_frames['mortality rate (%)'] = 100 * (data_frames['deaths'] / data_frames['confirmed cases']).fillna(0)

# Countries to show
countries = list(argv[1:]) or ['Netherlands', 'Germany', 'Italy', 'Spain', 'France', 'Belgium', 'Poland', 'Czechia',
                               'Lithuania', 'United Kingdom', 'US', 'Taiwan*', 'Singapore', 'Korea, South', 'India',
                               'China', 'Rest', 'World']

# Plot
cf.go_offline()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
charts = [
    data_frames[chart][countries].iplot(
        asFigure=True,
        title='Wuhan Corona Virus Pandemic ' + chart.title(),
        theme='solar',
        colors=['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55',
                '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7',
                '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72'],
    ) for chart in data_frames.keys()]
for chart in charts:
    chart.update_layout(hovermode='x', height=750)
app.layout = html.Div([dcc.Graph(figure=chart) for chart in charts])
app.title = 'Wuhan Corona Virus Pandemic'
app.run_server()
