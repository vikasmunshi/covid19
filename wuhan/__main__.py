#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
from datetime import datetime
from io import StringIO
from sys import argv
from time import time

import cufflinks as cf
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import requests
from flask import redirect, request

from .data_corrections import data_corrections
from .urls import urls


# retrieve data from URL and return Pandas DataFrame
def data_frame_from_url(url_name: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(requests.get(urls[url_name]).content.decode()))


# Population Data
def get_population():
    population_data = data_frame_from_url('population').set_index('Country').iloc[:, [-1]].dropna(0)
    population_data.columns = ['Population']
    population_data.Population = population_data.Population.astype(int)
    for country, name_change, missing_data in data_corrections:
        if name_change:
            population_data.loc[name_change] = population_data.loc[country]
        if missing_data:
            population_data.loc[country] = missing_data
    population_data = population_data / 10 ** 6
    population_data.loc['Rest'] = population_data.loc['World'] - population_data.loc['China']
    return population_data


# Retrieve Covid-19 Data and plot charts
def plot_covid19_data(population_data, countries_to_show):
    data_frames = {}
    for metric, df in ((k, data_frame_from_url(k)) for k in urls.keys() if k != 'population'):
        df = df.drop(0)[['Country/Region', 'Province/State', 'Date', 'Value']]
        df.Date = pd.to_datetime(df.Date)
        df.Value = df.Value.astype(int)
        df = df.groupby(['Country/Region', 'Date']).sum().unstack()
        df = pd.concat([df, population_data], axis=1, join='inner').drop('Population', axis=1)
        df = df.transpose()
        df.index = pd.MultiIndex.from_tuples(df.index)
        df = df.rename_axis(index=['Value', 'Date']).reset_index(['Value']).drop('Value', axis=1)
        df['World'] = df.sum(axis=1)
        df['Rest'] = df['World'] - df['China']
        n_days = 7
        delta = df.diff(n_days).fillna(0)
        reproduction_rate = ((delta / delta.shift(n_days)) ** (1 / n_days)).replace(np.inf, np.nan).fillna(0)
        data_frames[metric + ' reproduction rate'] = reproduction_rate
        data_frames[metric] = df
        data_frames['daily ' + metric] = df.diff(1).fillna(0)
        data_frames[metric + ' per million'] = per_million = df.div(population_data.Population, axis=1).fillna(0)
        data_frames['daily ' + metric + ' per million'] = per_million.diff(1).fillna(0)

    data_frames['case fatality rate (%)'] = 100 * (data_frames['deaths'] / data_frames['confirmed cases']).fillna(0)
    report_date = max(data_frames['case fatality rate (%)'].index).strftime('%d %b %Y')

    return [
        data_frames[metric][countries_to_show].iplot(
            asFigure=True,
            title='Wuhan Corona Virus Pandemic {} as on {} <i>retrieved {}</i>'.format(
                metric.title(), report_date, datetime.now().strftime('%d %b %Y %H:%M')),
            theme='solar',
            colors=['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616', '#479B55',
                    '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5', '#FF0092', '#22FFA7',
                    '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72'],
        ).update_layout(hovermode='x', height=750)
        for metric in data_frames.keys()
    ]


# Layout Charts, refresh every 12 hours
def create_layout(population_data, countries_to_show):
    def get_charts():
        return html.Div([
            dcc.Graph(figure=chart)
            for chart in plot_covid19_data(population_data=population_data, countries_to_show=countries_to_show)])

    cache = [[time(), get_charts()]]

    def layout():
        if time() - cache[0][0] > 12 * 3600:
            cache[0] = [time(), get_charts()]
        return cache[0][1]

    return layout


if __name__ == '__main__':
    # Countries to show and their population
    countries = list(argv[1:]) or ['Australia', 'Austria', 'Belgium', 'Brazil', 'China', 'Czechia', 'France', 'Germany',
                                   'India', 'Italy', 'Japan', 'Korea, South', 'Lithuania', 'Netherlands', 'Poland',
                                   'Singapore', 'Spain', 'Taiwan*', 'US', 'United Kingdom', 'World', 'Rest', ]
    population = get_population().loc[countries, :]

    # Dash
    cf.go_offline()
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


    @app.server.route('/shutdown')
    @app.server.route('/restart')
    def shutdown():
        global restart
        restart = request.path == '/restart'
        request.environ.get('werkzeug.server.shutdown')()
        return redirect('/', code=302)


    restart = True
    while restart:
        restart = False
        app.layout = create_layout(population_data=population, countries_to_show=countries)
        app.title = 'Wuhan Corona Virus Pandemic Stats'
        app.run_server(host='0.0.0.0')
