#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
from io import StringIO
from os import path
from sys import argv

import cufflinks as cf
import pandas as pd
import requests
import requests_cache

from .data_corrections import data_corrections
from .urls import urls

requests_cache.install_cache(path.split(path.split(__file__)[0])[1], expire_after=12 * 3600)


def data_frame_from_url(url_name: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(requests.get(urls[url_name]).content.decode()))


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
    data_frames[name + ' per million'] = data_frame.div(population.Population, axis=1).dropna(axis=1)
    data_frames[name + ' per week per million'] = data_frames[name + ' per million'].diff(7).dropna().astype(int)
data_frames['mortality rate (%)'] = 100 * (data_frames['deaths'] / data_frames['confirmed cases']).fillna(0)

countries = list(argv[1:])
cf.go_offline()
for chart in data_frames.keys():
    df = data_frames[chart][countries] if countries else data_frames[chart]
    fig = df.iplot(asFigure=True, title='Wuhan Corona Virus Pandemic ' + chart.title())
    fig['layout']['hovermode'] = 'x'
    fig.show()
