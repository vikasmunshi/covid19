#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
import datetime as dt
import inspect
import io
import math
import os
import platform
import random
import string
import sys
import threading
import time

import cufflinks as cf
import dash
import dash_core_components as dcc
import dash_html_components as dhc
import flask
import numpy as np
import pandas as pd
import plotly.express as px
import requests

__all__ = ['app', 'server']

auth_file = 'auth.txt'
if not os.path.exists(auth_file):
    with open(auth_file, 'w') as o_file:
        o_file.write('\n'.join([(''.join(random.choice(string.ascii_letters) for _ in range(128))) for n in range(99)]))
with open(auth_file) as in_file:
    auth_tokens = in_file.readlines()[42:69]
kill_payload = '/kill/' + auth_tokens[0]
reload_data_payload = '/reload/' + auth_tokens[-1]
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


def create_layout(keys: list = None, shown_keys: list = None, ds: str = '', ) -> {str: dhc.Div}:
    keys = ['Loading...'] if keys is None else keys
    shown_keys = keys[0] if shown_keys is None else shown_keys
    page_title = '{} {}'.format(app_title, ds)
    return {
        'Loading...': dhc.Div('Loading ...'),
        'layout': dhc.Div([
            dhc.H3([dhc.A(u'\u2388 ', href='/', style={'text-decoration': 'none'}, title='Refresh'), page_title]),
            dcc.Dropdown(id='chart', options=[{'label': k, 'value': k} for k in keys], value=shown_keys, multi=True),
            dhc.Div(id='page-content', style={'min-height': '600px'}),
            dhc.I([dhc.A(u'\u2317 ', title='Code', href='/code', target='_blank', style={'text-decoration': 'none'}),
                   'created: {}'.format(dt.datetime.now().strftime('%d %b %Y %H:%M')), ]), ]),
    }


cache = create_layout()
cache_loop_lock = threading.Lock()
cache_update_lock = threading.Lock()


def format_num(n: int) -> str:
    suffixes = ['', ' Thousand', ' Million', ' Billion']
    i = max(0, min(3, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return (str(i_n) if (i_n := int(n)) - n == 0 else '{:,.2f}'.format(n)) if i == 0 \
        else '{:,.3f}{}'.format(n / 10 ** (3 * i), suffixes[i])


# retrieve data from URL and return Pandas DataFrame
def data_frame_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(requests.get(url).content.decode()))


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
    df = df[['Country', 'Date', 'Code', 'Population', 'Cases', 'Deaths']]
    df = df.set_index(['Country', 'Date'])
    df['WeeklyCases'] = df.Cases.diff(7)
    df['WeeklyCases'][df['WeeklyCases'] < 0] = 0
    df['WeeklyDeaths'] = df.Deaths.diff(7)
    df['WeeklyDeaths'][df['WeeklyDeaths'] < 0] = 0
    df['CPM'] = 10 ** 6 * df.Cases / df.Population
    df['DPM'] = 10 ** 6 * df.Deaths / df.Population
    df['WeeklyDPM'] = 10 ** 6 * df.WeeklyDeaths / df.Population
    df['CFR'] = 100 * df.Deaths / df.Cases
    df['CRR'] = ((df.WeeklyCases / df.WeeklyCases.shift(7)) ** (1 / 7)).replace(np.inf, np.nan)
    return df


# Plot overview with country comparisons
def plot_comparision(df: pd.DataFrame, regions: list, last_date: dt.datetime) -> {str, dhc.Div}:
    # Plot single metric for select countries
    def plot_time_series(col: str, label: str, **kwargs) -> dcc.Graph:
        return dcc.Graph(figure=df[col].unstack().transpose()[regions].figure(title=label, **kwargs)
                         .update_layout(height=800, title_x=0.5, legend_orientation='h', hovermode='x'))

    df_current = df.xs(last_date, axis=0, level=1)
    df_geo = df_current.drop('World').reset_index()

    # Plot current value of single metric for every country
    def plot_current(col: str, label: str, drop_world: bool = False, cut_at: str = None, **kwargs) -> dcc.Graph:
        ds = df_current[col].drop('World') if drop_world else df_current[col]
        ds = ds.nlargest(42) if cut_at is None else ds[ds >= ds.loc[cut_at]]
        return dcc.Graph(figure=ds.sort_values().figure(title=label, kind='bar', orientation='h', **kwargs)
                         .update_layout(height=800, title_x=0.5, hovermode='y'))

    # Plot single metric for every country on a map
    def plot_geo(col: str, label: str, marker_color: str) -> dcc.Graph:
        return dcc.Graph(figure=px.scatter_geo(
            df_geo, projection='natural earth', title=label, locations='Code', size=col,
            hover_name='Country', hover_data=['Cases', 'Deaths', 'CPM', 'DPM', 'CFR'],
            color_discrete_sequence=[marker_color])
                         .update_layout(height=800, title_x=0.5)
                         .update_geos(resolution=50,
                                      showcountries=True, countrycolor='#663399',
                                      showcoastlines=True, coastlinecolor='#663399',
                                      showland=True, landcolor='#E3E3E3', showocean=True, oceancolor='#ADD8E6',
                                      showlakes=True, lakecolor='#ADD8E6', showrivers=True, rivercolor='#ADD8E6'))

    return {
        'Current Deaths': dhc.Div([chart for chart in [
            plot_current('DPM', 'Deaths Per Million', theme='polar', cut_at='World', color=['#C70039']),
            plot_current('Deaths', 'Deaths', theme='polar', drop_world=True, color=['#C70039']), ]]),
        'Maps Cases': dhc.Div([chart for chart in [
            plot_geo('Cases', 'Total Cases', '#4C33FF'),
            plot_geo('WeeklyCases', 'Last 7 Days Total Cases', '#4C33FF'), ]]),
        'Maps Cases Per Million': dhc.Div([chart for chart in [
            plot_geo('CPM', 'Cases Per Million', '#4C33FF'),
            plot_geo('WeeklyCPM', 'Last 7 Days Cases Per Million', '#4C33FF'), ]]),
        'Maps Deaths': dhc.Div([chart for chart in [
            plot_geo('Deaths', 'Total Deaths', '#C70039'),
            plot_geo('WeeklyDeaths', 'Last 7 Days Total Deaths', '#C70039'), ]]),
        'Maps Deaths Per Million': dhc.Div([chart for chart in [
            plot_geo('DPM', 'Deaths Per Million', '#C70039'),
            plot_geo('WeeklyDPM', 'Last 7 Days Deaths Per Million', '#C70039'), ]]),
        'Time-series Cases': dhc.Div([chart for chart in [
            plot_time_series('Cases', 'Total Cases', theme='polar'),
            plot_time_series('WeeklyCases', 'Weekly Cases (last 7 days)', theme='solar', kind='bar'),
            plot_time_series('CPM', 'Cases Per Million', theme='polar'),
            plot_time_series('WeeklyCPM', 'Weekly Cases (last 7 days) Per Million', theme='solar', kind='bar'), ]]),
        'Time-series Deaths': dhc.Div([chart for chart in [
            plot_time_series('Deaths', 'Total Deaths', theme='polar'),
            plot_time_series('WeeklyDeaths', 'Weekly Deaths (last 7 days)', theme='solar', kind='bar'),
            plot_time_series('DPM', 'Deaths Per Million', theme='polar'),
            plot_time_series('WeeklyDPM', 'Weekly Deaths (last 7 days) Per Million', theme='solar', kind='bar'), ]]),
        'Time-series Rates': dhc.Div([chart for chart in [
            plot_time_series('CFR', 'Case Fatality Rate (%)', theme='polar'),
            plot_time_series('CRR', 'Case Reproduction Rate (last 7 days average)', theme='polar', logy=True), ]]),
    }


# Plot regional charts
def plot_regions(df: pd.DataFrame, regions: list, last_date: dt.datetime) -> {str, dhc.Div}:
    columns_in_regional_chart, column_colors, column_titles = zip(
        ('Cases', '#4C33FF', 'Confirmed Cases'),
        ('WeeklyCases', '#4C33FF', 'Cases Last 7 Days'),
        ('CRR', '#FF00FF', 'Case Reproduction Rate (last 7 days average)'),
        ('Deaths', '#C70039', 'Attributed Deaths'),
        ('WeeklyDeaths', '#C70039', 'Deaths Last 7 Days'),
        ('CFR', '#C70039', 'Case Fatality Rate (%)'),
    )

    def plot_one(region: str) -> dhc.Div:
        p, c, d, dpm = [format_num(x) for x in df.loc[region].loc[last_date][['Population', 'Cases', 'Deaths', 'DPM']]]
        title = '<b>{}</b><BR><i>{} People, {} Cases, {} Deaths, {} Deaths Per Million, As On {}</i><BR>'.format(
            region, p, c, d, dpm, last_date.strftime('%d %b %Y'))
        return dhc.Div(dcc.Graph(
            figure=df.loc[region][list(columns_in_regional_chart)].figure(
                theme='polar', title=title, subplots=True, shape=(2, 3), legend=False,
                colors=column_colors, subplot_titles=column_titles
            ).update_layout(height=780, title_x=0.5, hovermode='x')))

    return {region: plot_one(region) for region in regions}


# Cufflinks
cf.go_offline()
# Dash
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.title = app_title
app.layout = lambda: cache['layout']
# Flask
server = app.server


@app.callback(dash.dependencies.Output('page-content', 'children'), [dash.dependencies.Input('chart', 'value')])
def update_output(value):
    return [cache[v] for v in value] if isinstance(value, list) else cache[value]


def update_cache() -> bool:
    if not cache_update_lock.locked():
        with cache_update_lock:
            print(dt.datetime.now(), 'Updating Cache', flush=True)
            cache['population'] = population = cache.get('population', get_population())
            df = transform_covid19_data(population)
            last_date = max(df.index.get_level_values(level=1))
            regions = list(df.xs(last_date, axis=0, level=1).sort_values(by='Deaths', ascending=False).index)
            short_list = regions[0:18] + ['Taiwan']
            cache.update(comparision_charts := plot_comparision(df, short_list, last_date))
            cache.update(plot_regions(df, regions, last_date))
            cache.update(create_layout(keys=list(sorted(comparision_charts.keys())) + regions, shown_keys=short_list,
                                       ds=last_date.strftime('%d %b %Y')))
            print(dt.datetime.now(), 'Cache Updated', flush=True)
        return True
    return False


# Refresh every 12th hour (43200 seconds) offset by 6 hrs (21600 - 7200 CET offset from UTC) i.e. 06:00, 18:00 CET
@server.before_first_request
def update_cache_in_background():
    def loop_update_cache():
        if not cache_loop_lock.locked():
            with cache_loop_lock:
                if platform.system() == 'Darwin':
                    __import__('caffeine')  # import has side-effects
                while True:
                    try:
                        update_cache()
                    except Exception as e:
                        print(dt.datetime.now(), 'Exception occurred updating cache\n', str(e), flush=True)
                        next_reload_data_at = time.time() + 3600
                    else:
                        next_reload_data_at = ((1 + (int(time.time()) // 43200)) * 43200) + 14400
                    while (wait := next_reload_data_at - int(time.time())) > 0:
                        time.sleep(wait / 2)

    if not cache_loop_lock.locked():
        threading.Thread(target=loop_update_cache, daemon=True).start()


@server.route('/status')
def status():
    return 'updating cache' if cache_update_lock.locked() else 'serving {} items from cache'.format(len(cache.keys()))


@server.route('/code')
def code():
    src = inspect.getsource(sys.modules[__name__]).splitlines()
    rows = len(src)
    cols = max(len(line) for line in src)
    return '{0}{1} rows={4} cols={5} {2}{3}{0}/{1}{2}'.format('<', 'textarea', '>', '\n'.join(src), rows, cols)


@server.route(reload_data_payload)
def reload_data():
    return 'Reloaded ...' if update_cache() else 'Reloading in progress ...'


@server.route(kill_payload)
def shutdown():
    cmd = flask.request.environ.get('werkzeug.server.shutdown')
    return 'Oops ...' if cmd is None else 'Killed' if cmd() is None else 'Hmmm ...'


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Plot Wuhan Corona Virus Covid-19 Impact around the World')
    parser.add_argument('-a', '--address', type=str, help='interface address, default 0.0.0.0 (127.0.0.1 with -d)')
    parser.add_argument('-p', '--port', type=int, help='interface port, default 8050 (8060 with -d)')
    parser.add_argument('-d', '--dev', action='store_true', help='use cached downloads only, default false')
    parser.add_argument('-k', '--kill', action='store_true', help='send kill payload to server')
    parser.add_argument('-r', '--reload', action='store_true', help='send reload data payload to server')
    parser.add_argument('-s', '--status', action='store_true', help='print server status')
    args = parser.parse_args()

    host = (args.address or '127.0.0.1') if args.dev else (args.address or '0.0.0.0')
    port = (args.port or 8060) if args.dev else (args.port or 8050)

    if args.kill:
        print(requests.get('http://{}:{}{}'.format(host, port, kill_payload)).content.decode())
    elif args.reload:
        print(requests.get('http://{}:{}{}'.format(host, port, reload_data_payload)).content.decode())
    elif args.status:
        print(requests.get('http://{}:{}/status'.format(host, port)).content.decode())
    else:
        if args.dev:
            __import__('requests_cache').install_cache('cache', expire_after=12 * 3600)
        app.run_server(host=host, port=port)
