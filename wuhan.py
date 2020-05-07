#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
from datetime import datetime
from io import StringIO
from threading import Thread
from time import sleep, time

import cufflinks as cf
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import requests

urls = {
    'population': 'https://datahub.io/JohnSnowLabs/population-figures-by-country/r/population-figures-by-country-csv.csv',
    'confirmed cases': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv',
    'deaths': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv',
    'recovered': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv',
}

data_corrections = [['Bahamas, The', 'Bahamas', None], ['Brunei Darussalam', 'Brunei', None],
                    ['Congo, Dem. Rep.', 'Congo (Kinshasa)', None], ['Congo, Rep.', 'Congo (Brazzaville)', None],
                    ['Czech Republic', 'Czechia', None], ['Egypt, Arab Rep.', 'Egypt', None],
                    ['Gambia, The', 'Gambia', None], ['Iran, Islamic Rep.', 'Iran', None],
                    ['Korea, Rep.', 'Korea, South', None], ['Kyrgyz Republic', 'Kyrgyzstan', None],
                    ['Lao PDR', 'Laos', None], ['Macedonia, FYR', 'North Macedonia', None], ['Myanmar', 'Burma', None],
                    ['Russian Federation', 'Russia', None], ['Slovak Republic', 'Slovakia', None],
                    ['Syrian Arab Republic', 'Syria', None], ['Taiwan*', None, 23780452], ['United States', 'US', None],
                    ['Venezuela, RB', 'Venezuela', None], ['Yemen, Rep.', 'Yemen', None]]


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

    return html.Div(
        [dcc.Graph(figure=chart) for chart in [
            data_frames[metric][countries_to_show].iplot(
                asFigure=True,
                title='Wuhan Corona Virus Pandemic {} as on {} <i>retrieved {}</i>'.format(
                    metric.title(), report_date, datetime.now().strftime('%d %b %Y %H:%M')),
                theme='solar',
                colors=['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616',
                        '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5',
                        '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72'],
            ).update_layout(hovermode='x', height=750) for metric in data_frames.keys()]])


# Layout Charts, refresh every 8th hour i.e. 00:00, 08:00, 16:00 (28800 seconds)
def create_layout(population_data, countries_to_show):
    cache = {'charts': html.Div('Retrieving Data...')}

    def update():
        while True:
            try:
                cache['charts'] = plot_covid19_data(population_data, countries_to_show)
                print(datetime.now(), 'Cache updated', flush=True)
            except Exception as e:
                print(datetime.now(), 'Exception occurred while updating cache\n', str(e), flush=True)
                next_update = (1 + int(time()) // 3600) * 3600
            else:
                next_update = (1 + int(time()) // 28800) * 28800
            while (diff := next_update - int(time())) > 0:
                print(datetime.now(), '{} seconds to next update'.format(diff), flush=True)
                sleep(diff // 2)

    Thread(target=update, daemon=True).start()

    return lambda: cache['charts']


# Countries to show and their population
countries = ['Australia', 'Austria', 'Belgium', 'Brazil', 'China', 'Czechia', 'France', 'Germany', 'India', 'Iran',
             'Italy', 'Japan', 'Korea, South', 'Lithuania', 'Netherlands', 'Poland', 'Singapore', 'Spain', 'Sweden',
             'Taiwan*', 'US', 'United Kingdom', 'World', 'Rest', ]
population = get_population().loc[countries, :]

# Dash
cf.go_offline()
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Wuhan Corona Virus Pandemic Stats'
app.layout = create_layout(population_data=population, countries_to_show=countries)
# Flask
server = app.server

if __name__ == '__main__':
    app.run_server()
