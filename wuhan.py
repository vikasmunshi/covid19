#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
from datetime import datetime
from io import StringIO
from math import floor, log10
from threading import Thread
from time import sleep, time

import cufflinks as cf
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import requests

__all__ = ['app', 'server', 'update_cache_in_background']

# noinspection SpellCheckingInspection
URLS = {
    'population': 'http://api.worldbank.org/countries/all/indicators/SP.POP.TOTL?format=csv',
    'cases': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest'
             '=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename'
             '-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename'
             '-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03'
             '=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on'
             '&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag'
             '=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header'
             '=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat'
             '&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent'
             '.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series'
             '%2Ftime_series_covid19_confirmed_global.csv',
    'deaths': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest'
              '=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename'
              '-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename'
              '-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03'
              '=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on'
              '&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag'
              '=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header'
              '=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat'
              '&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent'
              '.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series'
              '%2Ftime_series_covid19_deaths_global.csv',
    'recovered': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_narrow.csv'
                 '?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02'
                 '=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03'
                 '=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum'
                 '&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date'
                 '&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on'
                 '&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1'
                 '%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat'
                 '&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https'
                 '%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data'
                 '%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv',
}


def layout(title: str, keys: list, date_stamp: str = '', show_keys: list = None) -> html.Div:
    return html.Div([
        html.H6([
            html.A(u'\u2299', href='/', style={'text-decoration': 'none'}),
            ' {} {} (retrieved {})'.format(title, date_stamp, datetime.now().strftime('%d %b %H:%M'))]),
        dcc.Dropdown(id='region', options=[{'label': k, 'value': k} for k in keys],
                     value=keys[0] if show_keys is None else show_keys, multi=bool(show_keys)),
        html.Div(id='page-content')])


__cache__ = {'Loading...': html.Div('Retrieving Data ... '),
             'layout': layout(title='Wuhan Corona Virus Pandemic Stats', keys=['Loading...'])}


def format_num(n: int) -> str:
    suffixes = ['', ' Thousand', ' Million', ' Billion']
    i = max(0, min(3, int(floor(0 if n == 0 else log10(abs(n)) / 3))))
    return '{:,.3f}{}'.format(n / 10 ** (3 * i), suffixes[i])


# retrieve data from URL and return Pandas DataFrame
def data_frame_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(requests.get(url).content.decode()))


# Population Data
def get_population() -> pd.DataFrame:
    population = data_frame_from_url(URLS['population'])[['Country Name', 'Country Code', '2018']]
    population.columns = ['Country', 'Code', 'Population']
    for country, name_change, missing_data in [
        ['Bahamas, The', 'Bahamas', None], ['Brunei Darussalam', 'Brunei', None],
        ['Congo, Dem. Rep.', 'Congo (Kinshasa)', None], ['Congo, Rep.', 'Congo (Brazzaville)', None],
        ['Czech Republic', 'Czechia', None], ['Egypt, Arab Rep.', 'Egypt', None], ['Gambia, The', 'Gambia', None],
        ['Iran, Islamic Rep.', 'Iran', None], ['Korea, Rep.', 'Korea, South', None],
        ['Kyrgyz Republic', 'Kyrgyzstan', None], ['Lao PDR', 'Laos', None], ['Myanmar', 'Burma', None],
        ['Russian Federation', 'Russia', None], ['Slovak Republic', 'Slovakia', None],
        ['Syrian Arab Republic', 'Syria', None], ['United States', 'US', None],
        ['Venezuela, RB', 'Venezuela', None],
        ['Yemen, Rep.', 'Yemen', None], ['Taiwan', None, ['TWN', 23780452]]
    ]:
        if name_change:
            population.Country = population.Country.str.replace(pat=country, repl=name_change, regex=False)
        if missing_data:
            population.loc[len(population)] = [country] + missing_data
    return population


# Retrieve Covid-19 Data
def get_covid19_data(metric: str) -> pd.DataFrame:
    df = data_frame_from_url(URLS[metric]).drop(0)[['Country/Region', 'Province/State', 'Date', 'Value']]
    df.Date = pd.to_datetime(df.Date)
    df.Value = df.Value.astype(int)
    df = df.groupby(['Country/Region', 'Date']).sum().reset_index()
    df.columns = ['Country', 'Date', metric.title()]
    df.Country = df.Country.str.replace(pat='Taiwan*', repl='Taiwan', regex=False)
    return df.sort_values(by=['Country', 'Date'])


# Transform Covid-19 Data
def transform_covid19_data(population: pd.DataFrame) -> pd.DataFrame:
    df = get_covid19_data('cases')
    df = pd.merge(df, get_covid19_data('deaths'), on=['Country', 'Date'], how='inner')
    df = pd.concat([df, df.groupby('Date').sum().reset_index('Date').sort_values('Date')]).fillna('World')
    df = pd.merge(df, population, on=['Country'], how='left').dropna()
    df = df[['Country', 'Date', 'Population', 'Cases', 'Deaths']]
    df = df.set_index(['Country', 'Date'])
    df['WeeklyCases'] = df.Cases.diff(7)
    df['WeeklyDeaths'] = df.Deaths.diff(7)
    df[df < 0] = 0
    df['CPM'] = 10 ** 6 * df.Cases / df.Population
    df['DPM'] = 10 ** 6 * df.Deaths / df.Population
    df['CFR'] = 100 * df.Deaths / df.Cases
    df['CRR'] = ((df.WeeklyCases / df.WeeklyCases.shift(7)) ** (1 / 7)).replace(np.inf, np.nan)
    return df


# Plot overview with country comparisons
def plot_comparision(df: pd.DataFrame, countries_in_overview: list) -> {str, html.Div}:
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    styles = {'line': {'theme': 'polar', 'colors': colors * 4},
              'bar': {'kind': 'bar', 'theme': 'solar', 'colors': colors * 4}}

    # Plot single metric for select countries
    def plot_one(ds: pd.Series, label: str, style: str, **kwargs) -> dcc.Graph:
        return dcc.Graph(figure=ds.unstack().transpose()[countries_in_overview]
                         .figure(title=label, **styles[style], **kwargs)
                         .update_layout(height=800, title_x=0.5, legend_orientation='h', hovermode='x'))

    return {'Comparision': html.Div([c for c in [
        plot_one(df.Cases, 'Total Cases', 'line'),
        plot_one(df.CPM, 'Cases Per Million', 'line'),
        plot_one(df.WeeklyCases, 'Weekly Cases (last 7 days)', 'bar'),
        plot_one(df.Deaths, 'Total Deaths', 'line'),
        plot_one(df.DPM, 'Deaths Per Million', 'line'),
        plot_one(df.WeeklyDeaths, 'Weekly Deaths (last 7 days)', 'bar'),
        plot_one(df.CFR, 'Case Fatality Rate (%)', 'line'),
        plot_one(df.CRR, 'Case Reproduction Rate (last 7 days average)', 'line', logy=True)]])}


# Plot regional charts
def plot_regional(df: pd.DataFrame, regions_sorted_by_deaths: list, last_date: datetime) -> {str, html.Div}:
    columns_in_regional_chart = ['Cases', 'Deaths', 'WeeklyCases', 'WeeklyDeaths', 'CRR', 'CFR']
    column_titles = ['Confirmed Cases', 'Attributed Deaths', 'Cases Last 7 Days', 'Deaths Last 7 Days',
                     'Case Reproduction Rate (last 7 days average)', 'Case Fatality Rate (%)']
    column_colors = ['#0000FF', '#FF0000', '#0000FF', '#FF0000', '#FF00FF', '#FF0000']

    def plot_one(region: str) -> (str, html.Div):
        p, c, d, dpm = df.loc[region].loc[last_date][['Population', 'Cases', 'Deaths', 'DPM']]
        title = """
        <b>{}</b><BR>
        <b>{}</b> people 
        <b>{}</b> cases 
        <b>{}</b> deaths 
        <b>{:,.2f}</b> deaths per million 
        as on <b>{}</b><BR>
        """.format(region, format_num(p), format_num(c), format_num(d), dpm, last_date.strftime('%d %b %Y'))
        return region, html.Div(dcc.Graph(
            figure=df.loc[region][columns_in_regional_chart].figure(
                theme='polar', title=title, subplots=True, shape=(3, 2), legend=False,
                colors=column_colors, subplot_titles=column_titles
            ).update_layout(height=780, title_x=0.5)))

    return dict(plot_one(region) for region in regions_sorted_by_deaths)


# Refresh every 12th hour i.e. 00:00 UTC, 12:00 UTC (43200 seconds)
def update_cache_in_background() -> Thread:
    def update_cache():
        population = get_population()
        while True:
            try:
                df = transform_covid19_data(population)
                last_date = max(df.index.get_level_values(level=1))
                regions = list(df.xs(last_date, axis=0, level=1).sort_values(by='Deaths', ascending=False).index)
                short_list = regions[0:31] + ['Taiwan']
                __cache__.update(plot_comparision(df, short_list))
                __cache__.update(plot_regional(df, regions, last_date))
                __cache__['layout'] = layout(title='Wuhan Corona Virus Pandemic stats', keys=['Comparision'] + regions,
                                             date_stamp=last_date.strftime('%d %b %Y'), show_keys=short_list)
                print(datetime.now(), 'Cache Updated', flush=True)
            except Exception as e:
                print(datetime.now(), 'Exception occurred while updating cache\n', str(e), flush=True)
                at = (1 + int(time()) // 3600) * 3600
            else:
                at = (1 + int(time()) // 43200) * 43200
            while (wait := at - int(time())) > 0:
                print(datetime.now(), 'Next Cache Update at {}'.format(datetime.utcfromtimestamp(at)), flush=True)
                sleep(min(wait / 2, 3000))

    thread = Thread(target=update_cache, daemon=True)
    thread.start()
    return thread


# Cufflinks
cf.go_offline()
# Dash
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = 'Wuhan Corona Virus Pandemic Stats'
app.layout = lambda: __cache__['layout']
# Flask
server = app.server


@app.callback(dash.dependencies.Output('page-content', 'children'), [dash.dependencies.Input('region', 'value')])
def update_output(value):
    return [__cache__[v] for v in value] if isinstance(value, list) else __cache__[value]


if __name__ == '__main__':
    if 'dev' in __import__('sys').argv:
        __import__('requests_cache').install_cache('cache', expire_after=12 * 3600)

    update_cache_in_background()
    app.run_server(host='0.0.0.0')
