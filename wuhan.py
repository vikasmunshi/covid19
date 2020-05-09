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
    'Population': 'https://datahub.io/JohnSnowLabs/population-figures-by-country/r/population-figures-by-country-csv.csv',
    'Cases': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv',
    'Deaths': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv',
    'Recovered': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv',
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
    population_data = data_frame_from_url('Population').set_index('Country').loc[:, ['Year_2016']].dropna(0)
    population_data.columns = ['Population']
    population_data.Population = population_data.Population.astype(int)
    for country, name_change, missing_data in data_corrections:
        if name_change:
            population_data.loc[name_change] = population_data.loc[country]
        if missing_data:
            population_data.loc[country] = missing_data
    population_data = population_data / 10 ** 6
    return population_data


# Retrieve Covid-19 Data and plot charts
def plot_covid19_data(population_data):
    params = {'theme': 'solar',
              'colors': ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616',
                         '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5',
                         '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']}
    title = 'Wuhan Corona Virus Pandemic {}{}{} as on {} <i>Retrieved {}</i>'
    charts = []
    cfr = last_date = countries_to_show = None
    retrieved = datetime.now().strftime('%d %b %Y %H:%M')
    for metric, df in ((k, data_frame_from_url(k)) for k in ('Deaths', 'Cases')):
        df = df.drop(0)[['Country/Region', 'Province/State', 'Date', 'Value']]
        df.Date = pd.to_datetime(df.Date)
        df.Value = df.Value.astype(int)
        df = df.groupby(['Country/Region', 'Date']).sum().unstack().transpose()
        df.index = pd.MultiIndex.from_tuples(df.index)
        df = df.rename_axis(index=['Value', 'Date']).reset_index(['Value'], drop=True)
        df['World'] = df.sum(axis=1)
        per_mil = df.div(population_data.Population, axis=1).fillna(0)

        if metric == 'Deaths':
            last_date = max(df.index)
            countries_to_show = list(df.loc[last_date].nlargest(31).index) + ['Taiwan*']
            countries_to_show = per_mil[countries_to_show].loc[last_date].sort_values(ascending=False).index
            last_date = last_date.strftime('%d %b %Y')
            cfr = df[countries_to_show]

        n_days = 7
        df = df[countries_to_show]
        delta = df.diff(n_days).fillna(0)
        daily = delta / n_days
        per_mil = per_mil[countries_to_show]
        dpm = per_mil.diff(n_days).fillna(0) / n_days

        daily_label = 'Daily ({} day mean) '.format(n_days)
        t = title.format('{}', metric, '{}', last_date, retrieved)

        charts.append(dpm.figure(title=t.format(daily_label, ' per Million'), kind='bar', **params))
        charts.append(per_mil.figure(title=t.format('', ' per Million'), **params))
        charts.append(daily.figure(title=t.format(daily_label, ''), kind='bar', **params))
        charts.append(df.figure(title=t.format('Total ', ''), **params))

        if metric == 'Cases':
            rr = ((delta / delta.shift(n_days)) ** (1 / n_days)).replace(np.inf, np.nan).fillna(0)
            cfr = 100 * cfr / df
            charts.append(rr.figure(title=t.format('Case Reproduction Rate (', ')'), **params))
            charts.append(cfr.figure(title=title.format('Case Fatality Rate', '', '', last_date, retrieved), **params))

    return html.Div([dcc.Graph(figure=chart.update_layout(hovermode='x', height=800)) for chart in reversed(charts)])


# Layout Charts, refresh every 12th hour i.e. 00:00 UTC, 12:00 UTC (43200 seconds)
def create_layout(population_data):
    cache = {'charts': html.Div('Retrieving Data...')}

    def update_cache():
        while True:
            try:
                cache['charts'] = plot_covid19_data(population_data)
                print(datetime.now(), 'Cache updated', flush=True)
            except Exception as e:
                print(datetime.now(), 'Exception occurred while updating cache\n', str(e), flush=True)
                update_at = (1 + int(time()) // 3600) * 3600
            else:
                update_at = (1 + int(time()) // 43200) * 43200
            while (diff := update_at - time()) > 0:
                print(datetime.now(), '{}s to update at {}'.format(diff, datetime.fromtimestamp(update_at)), flush=True)
                sleep(diff / 2)

    Thread(target=update_cache, daemon=True).start()

    return lambda: cache['charts']


# Cufflinks
cf.go_offline()
# Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Wuhan Corona Virus Pandemic Stats'
app.layout = create_layout(population_data=get_population())
# Flask
server = app.server

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
