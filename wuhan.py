#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
from datetime import datetime
from io import StringIO
from math import floor, log10
from threading import Lock, Thread
from time import sleep, time

import cufflinks as cf
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import requests

__all__ = ['app', 'server']
# noinspection SpellCheckingInspection
__kill__ = '/kill/fbhEGrxFzMpHhQAcsiAmnCZTFeROcstAxcpAMvSJIQnRwZRNFbXsZpqScLMnRbEk'
# noinspection SpellCheckingInspection
__restart__ = '/restart/bnboeqzAigIRGYzKghFZhzCDdxRGPiLlYATXkSdpSlrRQRmnCEFxrZiXMDYqgNlU'
app_title = 'Wuhan Corona Virus Pandemic Stats'
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


def create_layout(title: str, keys: list, date_stamp: str = '', show_keys: list = None) -> html.Div:
    return html.Div([
        html.H6(['{} {} (retrieved {})'.format(title, date_stamp, datetime.now().strftime('%d %b %H:%M')),
                 html.A(u' \u229B', title='Reload Page', href='/', style={'text-decoration': 'none'})]),
        dcc.Dropdown(id='region', options=[{'label': k, 'value': k} for k in keys],
                     value=keys[0] if show_keys is None else show_keys, multi=bool(show_keys)),
        html.Div(id='page-content')])


__cache__ = {'Loading...': html.Div('Retrieving Data ... '),
             'layout': create_layout(title=app_title, keys=['Loading...'])}
__cache_lock__ = Lock()


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
    df['WeeklyDPM'] = 10 ** 6 * df.WeeklyDeaths / df.Population
    df['CFR'] = 100 * df.Deaths / df.Cases
    df['CRR'] = ((df.WeeklyCases / df.WeeklyCases.shift(7)) ** (1 / 7)).replace(np.inf, np.nan)
    return df


# Plot overview with country comparisons
def plot_comparision(df: pd.DataFrame, countries_in_overview: list) -> {str, html.Div}:
    # Plot single metric for select countries
    def plot_one(ds: pd.Series, label: str, **kwargs) -> dcc.Graph:
        return dcc.Graph(figure=ds.unstack().transpose()[countries_in_overview].figure(title=label, **kwargs)
                         .update_layout(height=800, title_x=0.5, legend_orientation='h', hovermode='x'))

    return {'Comparision': html.Div([chart for chart in [
        plot_one(df.Cases, 'Total Cases', theme='polar'),
        plot_one(df.CPM, 'Cases Per Million', theme='polar'),
        plot_one(df.WeeklyCases, 'Weekly Cases (last 7 days)', theme='solar', kind='bar'),
        plot_one(df.Deaths, 'Total Deaths', theme='polar'),
        plot_one(df.DPM, 'Deaths Per Million', theme='polar'),
        plot_one(df.WeeklyDeaths, 'Weekly Deaths (last 7 days)', theme='solar', kind='bar'),
        plot_one(df.WeeklyDPM, 'Weekly Deaths (last 7 days) Per Million', theme='solar', kind='bar'),
        plot_one(df.CFR, 'Case Fatality Rate (%)', theme='polar'),
        plot_one(df.CRR, 'Case Reproduction Rate (last 7 days average)', theme='polar', logy=True), ]])}


# Plot regional charts
def plot_regions(df: pd.DataFrame, regions_sorted_by_deaths: list, last_date: datetime) -> {str, html.Div}:
    columns_in_regional_chart, column_colors, column_titles = zip(
        ('Cases', '#0000FF', 'Confirmed Cases'),
        ('WeeklyCases', '#0000FF', 'Cases Last 7 Days'),
        ('CRR', '#FF00FF', 'Case Reproduction Rate (last 7 days average)'),
        ('Deaths', '#FF0000', 'Attributed Deaths'),
        ('WeeklyDeaths', '#FF0000', 'Deaths Last 7 Days'),
        ('CFR', '#FF0000', 'Case Fatality Rate (%)'),
    )

    def plot_one(region: str) -> html.Div:
        p, c, d, dpm = df.loc[region].loc[last_date][['Population', 'Cases', 'Deaths', 'DPM']]
        title = '<b>{}</b><BR><i>{} People, {} Cases, {} Deaths, {:,.2f} Deaths Per Million, As On {}</i><BR>' \
            .format(region, format_num(p), format_num(c), format_num(d), dpm, last_date.strftime('%d %b %Y'))
        return html.Div(dcc.Graph(
            figure=df.loc[region][list(columns_in_regional_chart)].figure(
                theme='polar', title=title, subplots=True, shape=(2, 3), legend=False,
                colors=column_colors, subplot_titles=column_titles
            ).update_layout(height=780, title_x=0.5)))

    return {region: plot_one(region) for region in regions_sorted_by_deaths}


# Cufflinks
cf.go_offline()
# Dash
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = app_title
app.layout = lambda: __cache__['layout']
# Flask
server = app.server


@app.callback(dash.dependencies.Output('page-content', 'children'), [dash.dependencies.Input('region', 'value')])
def update_output(value):
    return [__cache__[v] for v in value] if isinstance(value, list) else __cache__[value]


# Refresh every 12th hour i.e. 00:00 UTC, 12:00 UTC (43200 seconds)
@server.before_first_request
def update_cache_in_background():
    def update_cache():
        __cache_lock__.acquire()
        population = get_population()
        while __cache_lock__.locked():
            try:
                print(datetime.now(), 'Updating Cache', flush=True)
                df = transform_covid19_data(population)
                last_date = max(df.index.get_level_values(level=1))
                regions = list(df.xs(last_date, axis=0, level=1).sort_values(by='Deaths', ascending=False).index)
                short_list = regions[0:31] + ['Taiwan']
                __cache__.update(plot_comparision(df, short_list))
                __cache__.update(plot_regions(df, regions, last_date))
                __cache__['layout'] = create_layout(title=app_title, keys=['Comparision'] + regions,
                                                    date_stamp=last_date.strftime('%d %b %Y'), show_keys=short_list)
                print(datetime.now(), 'Cache Updated', flush=True)
            except Exception as e:
                print(datetime.now(), 'Exception occurred while updating cache\n', str(e), flush=True)
                at = (1 + int(time()) // 3600) * 3600
            else:
                at = (1 + int(time()) // 43200) * 43200
            while (wait := at - int(time())) > 0:
                print(datetime.now(), 'Next Cache Update at {}'.format(datetime.utcfromtimestamp(at)), flush=True)
                sleep(min(wait / 2, 600))
        __cache_lock__.release()

    if not __cache_lock__.locked():
        Thread(target=update_cache, daemon=True).start()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot Wuhan Corona Virus Covid-19 Impact around the World')
    # noinspection SpellCheckingInspection
    parser.add_argument('-a', '--addr', type=str, help='interface address, default 0.0.0.0 (127.0.0.1 with -d)')
    parser.add_argument('-p', '--port', type=int, help='interface port, default 8050 (8060 with -d)')
    parser.add_argument('-d', '--dev', action='store_true', help='use cached downloads only, default false')
    parser.add_argument('-s', '--stop', action='store_true', help='send kill payload to running server')
    parser.add_argument('-r', '--restart', action='store_true', help='send restart payload to running server')
    args = parser.parse_args()

    host = (args.addr or '127.0.0.1') if args.dev else (args.addr or '0.0.0.0')
    port = (args.port or 8060) if args.dev else (args.port or 8050)

    if args.stop:
        print(requests.get('http://127.0.0.1:{}{}'.format(port, __kill__)).content.decode())
    elif args.restart:
        print(requests.get('http://127.0.0.1:{}{}'.format(port, __restart__)).content.decode())
    else:
        if args.dev:
            __import__('requests_cache').install_cache('cache', expire_after=12 * 3600)
        __start_server__ = True


        @server.route(__restart__)
        @server.route(__kill__)
        def shutdown():
            global __start_server__
            from flask import request
            __start_server__ = request.path == __restart__
            request.environ.get('werkzeug.server.shutdown')()
            return 'Restarted' if __start_server__ else 'Killed'


        while __start_server__:
            __start_server__ = False
            app.run_server(host=host, port=port)
