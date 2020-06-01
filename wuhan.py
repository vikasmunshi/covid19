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

__all__ = ['app', 'server', 'client']

auth_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auth.txt')
if not os.path.exists(auth_file):
    with open(auth_file, 'w') as o_file:
        o_file.write('\n'.join([(''.join(random.choice(string.ascii_letters) for _ in range(128))) for n in range(99)]))
with open(auth_file) as in_file:
    auth_tokens = in_file.readlines()[42:69]
code_link = '/code'
status_link = '/status'
kill_link = '/kill/' + auth_tokens[0]
reload_data_link = '/reload/' + auth_tokens[-1]
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
    keys = keys or ['Loading...']
    shown_keys = shown_keys or keys[0]
    page_title = '{} {}'.format(app_title, ds)
    report_date = dt.datetime.now().strftime('%H:%M %d-%b-%Y')
    link_style = {'text-decoration': 'none'}
    return {
        'report_date': report_date,
        'Loading...': dhc.Div('Loading ...'),
        'layout': dhc.Div([
            dhc.H3([dhc.A(u'\u2388 ', href='/', style=link_style, title='Refresh'), page_title]),
            dcc.Dropdown(id='chart', options=[{'label': k, 'value': k} for k in keys], value=shown_keys, multi=True),
            dhc.Div(id='page-content', style={'min-height': '600px'}),
            dhc.I([dhc.A(u'\u2317 ', title='Code', href=code_link, target='_blank', style=link_style),
                   'created: {}'.format(report_date), ]), ]), }


cache = create_layout()
cache_loop_lock = threading.Lock()
cache_update_lock = threading.Lock()


def format_num(n: float) -> str:
    suffixes = ['', ' Thousand', ' Million', ' Billion']
    i = max(0, min(3, int(math.floor(0 if n == 0 else math.log10(abs(n)) / 3))))
    return ('{:,.0f}'.format(n) if int(n) == n else '{:,.2f}'.format(n)) if i <= 1 \
        else '{:,.3f}{}'.format(n / 10 ** (3 * i), suffixes[i])


def log_message(*msg: str) -> None:
    print('- - -', dt.datetime.now().strftime('[%d/%b/%Y %H:%M:%S]'), ' '.join(msg), flush=True)


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
    df['WeeklyCases'][df['WeeklyCases'] < 0] = np.nan
    df['CPM'] = 10 ** 6 * df.Cases / df.Population
    df['WeeklyCPM'] = 10 ** 6 * df.WeeklyCases / df.Population
    df['DailyCPM'] = df.WeeklyCPM / 7
    df['DailyRateCPM'] = df.DailyCPM.diff(7) / 7

    df['WeeklyDeaths'] = df.Deaths.diff(7)
    df['WeeklyDeaths'][df['WeeklyDeaths'] < 0] = np.nan
    df['DPM'] = 10 ** 6 * df.Deaths / df.Population
    df['WeeklyDPM'] = 10 ** 6 * df.WeeklyDeaths / df.Population
    df['DailyDPM'] = df.WeeklyDPM / 7
    df['DailyRateDPM'] = df.DailyDPM.diff(7) / 7

    df['CFR'] = 100 * df.Deaths / df.Cases
    df['CRR'] = ((df.WeeklyCases / df.WeeklyCases.shift(7)) ** (1 / 7)).replace(np.inf, np.nan)
    df['DRR'] = ((df.WeeklyDeaths / df.WeeklyDeaths.shift(7)) ** (1 / 7)).replace(np.inf, np.nan)

    return df


# Plot overview with country comparisons
def plot_comparision(df: pd.DataFrame, regions: list, last_date: dt.datetime) -> {str, dhc.Div}:
    # Plot single metric for select countries
    def plot_time_series(col: str, label: str, **kwargs) -> dcc.Graph:
        return dcc.Graph(figure=df[col].unstack().transpose()[regions].figure(title=label, **kwargs)
                         .update_layout(height=800, title_x=0.5, legend_orientation='h', hovermode='x'))

    df_current = df.xs(last_date, axis=0, level=1).drop('World').fillna(0).reset_index()
    rag_scale = [(0.0, 'green'), (0.015625, 'blue'), (0.0625, 'yellow'), (0.25, 'orange'), (1.0, 'red')]

    # Plot current value of single metric for every country
    def plot_current(col: str, label: str, **kwargs) -> dcc.Graph:
        ds = df_current[['Country', col]].nlargest(42, columns=col).sort_values(by=col)
        return dcc.Graph(figure=ds.figure(title=label, x='Country', y=col, kind='bar', orientation='h', **kwargs)
                         .update_layout(height=800, title_x=0.5, hovermode='y'))

    # Plot Scatter of current values of two metrics for every country
    def plot_scatter(x: str, y: str, label: str) -> dcc.Graph:
        return dcc.Graph(figure=px.scatter(df_current, title=label, x=x, y=y, hover_name='Country')
                         .update_layout(height=800, title_x=0.5))

    # Plot single metric for every country on a map
    def plot_geo(col: str, label: str, color_countries: bool, colors: list) -> dcc.Graph:
        if color_countries:
            plotter, plotter_args = px.choropleth, {'color': col, 'color_continuous_scale': colors}
        else:
            plotter, plotter_args = px.scatter_geo, {'size': col, 'color_discrete_sequence': colors}
        return dcc.Graph(figure=plotter(df_current, title=label, height=800, locations='Code', hover_name='Country',
                                        hover_data=['Cases', 'Deaths', 'CPM', 'DPM', 'CFR'], **plotter_args)
                         .update_layout(title_x=0.5)
                         .update_geos(resolution=50,
                                      showcountries=True, countrycolor='#663399',
                                      showcoastlines=True, coastlinecolor='#663399',
                                      showland=True, landcolor='#E3E3E3', showocean=True, oceancolor='#ADD8E6',
                                      showlakes=True, lakecolor='#ADD8E6', showrivers=True, rivercolor='#ADD8E6'))

    return {
        'Scatter': dhc.Div([chart for chart in [
            plot_scatter(x='Deaths', y='Cases', label='Deaths vs Cases'),
            plot_scatter(x='DPM', y='CPM', label='Deaths/Million vs Cases/Million'),
            plot_scatter(x='DPM', y='CFR', label='Deaths/Million vs Case Fatality Rate'), ]]),
        'Current Cases': dhc.Div([chart for chart in [
            plot_current(col='Cases', label='Cases', theme='polar', color=['#4C33FF']),
            plot_current(col='CPM', label='Cases Per Million', theme='polar', color=['#4C33FF']), ]]),
        'Current Deaths': dhc.Div([chart for chart in [
            plot_current(col='Deaths', label='Deaths', theme='polar', color=['#C70039']),
            plot_current(col='DPM', label='Deaths Per Million', theme='polar', color=['#C70039']), ]]),
        'Maps Cases': dhc.Div([chart for chart in [
            plot_geo(col='Cases', label='Total Cases', color_countries=True, colors=rag_scale),
            plot_geo(col='WeeklyCases', label='Last Week Total Cases', color_countries=True, colors=rag_scale),
            plot_geo(col='CPM', label='Cases/Million', color_countries=False, colors=['#4C33FF']),
            plot_geo(col='WeeklyCPM', label='Last Week Cases/Million', color_countries=False, colors=['#4C33FF']), ]]),
        'Maps Deaths': dhc.Div([chart for chart in [
            plot_geo(col='Deaths', label='Total Deaths', color_countries=True, colors=rag_scale),
            plot_geo(col='WeeklyDeaths', label='Last Week Total Deaths', color_countries=True, colors=rag_scale),
            plot_geo(col='DPM', label='Deaths/Million', color_countries=False, colors=['#C70039']),
            plot_geo(col='WeeklyDPM', label='Last Week Deaths/Million', color_countries=False, colors=['#C70039']), ]]),
        'Time-series Cases': dhc.Div([chart for chart in [
            plot_time_series(col='Cases', label='Total Cases', theme='polar'),
            plot_time_series(col='WeeklyCases', label='Weekly Cases (last 7 days)', theme='solar', kind='bar'),
            plot_time_series(col='CPM', label='Cases Per Million', theme='polar'),
            plot_time_series(col='WeeklyCPM', label='Weekly Cases/Million', theme='solar', kind='bar'), ]]),
        'Time-series Deaths': dhc.Div([chart for chart in [
            plot_time_series(col='Deaths', label='Total Deaths', theme='polar'),
            plot_time_series(col='WeeklyDeaths', label='Weekly Deaths (last 7 days)', theme='solar', kind='bar'),
            plot_time_series(col='DPM', label='Deaths Per Million', theme='polar'),
            plot_time_series(col='WeeklyDPM', label='Weekly Deaths/Million', theme='solar', kind='bar'), ]]),
        'Time-series Rates': dhc.Div([chart for chart in [
            plot_time_series(col='CFR', label='Case Fatality Rate (%)', theme='polar'),
            plot_time_series(col='CRR', label='7 Day Mean Reproduction Rate - Cases', theme='polar', logy=True),
            plot_time_series(col='DRR', label='7 Day Mean Reproduction Rate - Deaths', theme='polar', logy=True), ]]),
    }


# Plot regional charts
def plot_regions(df: pd.DataFrame, regions: list, last_date: dt.datetime) -> {str, dhc.Div}:
    columns_in_chart, column_colors, column_titles = zip(
        ('CPM', '#4C33FF', 'Cases/Million'),
        ('DPM', '#C70039', 'Deaths/Million'),
        ('Cases', '#4C33FF', 'Total Cases'),
        ('DailyCPM', '#4C33FF', 'Cases/Day/Million (7 day average)'),
        ('DailyDPM', '#C70039', 'Deaths/Day/Million (7 day average)'),
        ('Deaths', '#C70039', 'Attributed Deaths'),
        ('DailyRateCPM', '#4C33FF', 'Growth Cases/Day/Million (7 day average)'),
        ('DailyRateDPM', '#C70039', 'Growth Deaths/Day/Million (7 day average)'),
        ('CFR', '#FF00FF', 'Case Fatality Rate (%)'),
    )
    summary_columns = ['Population', 'Cases', 'Deaths', 'DPM', 'CFR']

    def plot_region(region: str) -> dhc.Div:
        summary_values = [format_num(x) for x in df.loc[region].loc[last_date][summary_columns]]
        title = '<b>{}</b><BR>{} People, {} Cases, {} Deaths, {} Deaths/Million, {}% Case Fatality Rate' \
                '<i> as on {}</i><BR><BR>'.format(region, *summary_values, last_date.strftime('%d %b %Y'))
        return dhc.Div(dcc.Graph(figure=df.loc[region][list(columns_in_chart)]
                                 .figure(theme='polar', title=title, subplots=True, shape=(3, 3), legend=False,
                                         colors=column_colors, subplot_titles=column_titles)
                                 .update_layout(height=800, title_x=0.5, hovermode='x')))

    return {region: plot_region(region) for region in regions}


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
            log_message('Updating Cache')
            cache['population'] = population = cache.get('population', get_population())
            df = transform_covid19_data(population)
            last_date = max(df.index.get_level_values(level=1))
            regions = list(df.xs(last_date, axis=0, level=1).sort_values(by='Deaths', ascending=False).index)
            short_list = regions[0:32] + ['Taiwan']
            cache.update(comparision_charts := plot_comparision(df, short_list, last_date))
            cache.update(plot_regions(df, regions, last_date))
            cache.update(create_layout(keys=list(sorted(comparision_charts.keys())) + regions, shown_keys=short_list,
                                       ds=last_date.strftime('%d %b %Y')))
            log_message('Cache Updated')
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
                        log_message('Exception occurred while updating cache\n', str(e))
                        next_reload_data_at = time.time() + 3600
                    else:
                        next_reload_data_at = ((1 + (int(time.time()) // 43200)) * 43200) + 14400
                    while (wait := next_reload_data_at - int(time.time())) > 0:
                        time.sleep(wait / 2)

    if not cache_loop_lock.locked():
        threading.Thread(target=loop_update_cache, daemon=True).start()


@server.route(status_link)
def status():
    return 'updating cache' if cache_update_lock.locked() \
        else 'serving {} items cached at {}'.format(len(cache), cache['report_date'])


@server.route(code_link)
def code():
    src = inspect.getsource(sys.modules[__name__]).splitlines()
    rows = len(src)
    cols = max(len(line) for line in src)
    return '{0}{1} rows={4} cols={5} {2}{3}{0}/{1}{2}'.format('<', 'textarea', '>', '\n'.join(src), rows, cols)


@server.route(reload_data_link)
def reload_data():
    return 'Reloaded ...' if update_cache() else 'Reloading in progress ...'


@server.route(kill_link)
def shutdown():
    cmd = flask.request.environ.get('werkzeug.server.shutdown')
    return 'Oops ...' if cmd is None else 'Killed' if cmd() is None else 'Hmmm ...'


def client(host_address, client_port, payload) -> str:
    try:
        return requests.get('http://{}:{}{}'.format(host_address, client_port, payload)).content.decode()
    except requests.exceptions.ConnectionError:
        return ''


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

    host = args.address or ('127.0.0.1' if args.dev else '0.0.0.0')
    port = args.port or (8060 if args.dev else 8050)
    client_cmd = kill_link if args.kill else reload_data_link if args.reload else status_link if args.status else ''

    if client_cmd:
        if response := client(host_address=host, client_port=port, payload=client_cmd):
            log_message(response)
        else:
            log_message('http://{}:{} is down'.format(host, port))
            exit(1)
    else:
        if args.dev:
            __import__('requests_cache').install_cache('cache', expire_after=12 * 3600)
        app.run_server(host=host, port=port)
