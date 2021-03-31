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

auth_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auth_key')
if not os.path.exists(auth_file):
    with open(auth_file, 'w') as o_f:
        o_f.write('\n'.join([(''.join(random.choice(string.ascii_letters) for _ in range(256))) for n in range(256)]))
with open(auth_file) as in_file:
    auth_tokens = in_file.readlines()[42:69]
code_link = '/code'
ld_link = '/logs'
ld_file = None
status_link = '/status'
kill_link = '/kill/' + auth_tokens[0]
reload_data_link = '/reload/' + auth_tokens[-1]
app_title = 'Wuhan Corona Virus Pandemic Stats'
# noinspection SpellCheckingInspection
URLS = {
    'population': 'https://datahub.io/JohnSnowLabs/population-figures-by-country/r/population-figures-by-country-csv'
                  '.csv',
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
EU = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France',
      'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands',
      'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']


def create_layout(keys: list = None, shown_keys: list = None, ds: str = '', ) -> {str: dhc.Div}:
    keys = keys or ['Loading...']
    shown_keys = shown_keys or keys[0]
    page_title = '{} {}'.format(app_title, ds)
    report_date = dt.datetime.now().strftime('%H:%M %d-%b-%Y')
    link_style = {'text-decoration': 'none'}
    log_link = dhc.A(u'\u2317 ', title='Logs', href=ld_link, target='_blank', style=link_style) if ld_file else ''
    return {
        'report_date': report_date,
        'Loading...': dhc.Div('Loading ...'),
        'layout': dhc.Div([
            dhc.H3([dhc.A(u'\u2388 ', href='/', style=link_style, title='Refresh'), page_title]),
            dcc.Dropdown(id='chart', options=[{'label': k, 'value': k} for k in keys], value=shown_keys, multi=True),
            dhc.Div(id='page-content', style={'min-height': '600px'}),
            dhc.I([log_link, dhc.A(u'\u2318 ', title='Code', href=code_link, target='_blank', style=link_style),
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
    try:
        return pd.read_csv(io.StringIO(requests.get(url).content.decode()))
    except Exception as e:
        log_message('error retrieving url {}'.format(url))
        raise e
    finally:
        log_message('retrieved url {}'.format(url))


# Population Data
def get_population() -> pd.DataFrame:
    population = data_frame_from_url(URLS['population'])[['Country', 'Country_Code', 'Year_2016']]
    population.columns = ['Country', 'Code', 'Population']
    for country, name_change, missing_data in [
        ['Bahamas, The', 'Bahamas', None], ['Brunei Darussalam', 'Brunei', None],
        ['Congo, Dem. Rep.', 'Congo (Kinshasa)', None], ['Congo, Rep.', 'Congo (Brazzaville)', None],
        ['Czech Republic', 'Czechia', None], ['Egypt, Arab Rep.', 'Egypt', None], ['Gambia, The', 'Gambia', None],
        ['Iran, Islamic Rep.', 'Iran', None], ['Korea, Rep.', 'Korea, South', None],
        ['Kyrgyz Republic', 'Kyrgyzstan', None], ['Lao PDR', 'Laos', None], ['Macedonia, FYR', 'North Macedonia', None],
        ['Myanmar', 'Burma', None], ['Russian Federation', 'Russia', None], ['Slovak Republic', 'Slovakia', None],
        ['Syrian Arab Republic', 'Syria', None], ['Venezuela, RB', 'Venezuela', None], ['Yemen, Rep.', 'Yemen', None],
        ['Taiwan', None, ['TWN', 23780452]]
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
    df.Country = df.Country \
        .str.replace(pat='Taiwan*', repl='Taiwan', regex=False) \
        .str.replace(pat='US', repl='United States', regex=False)
    return df.sort_values(by=['Country', 'Date'])


# Transform Covid-19 Data
def transform_covid19_data(population: pd.DataFrame, cases: pd.DataFrame, deaths: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(cases, deaths, on=['Country', 'Date'], how='inner')
    eu = df[df.Country.isin(EU)].groupby('Date').sum().reset_index('Date').sort_values('Date')
    eu['Country'] = 'European Union'
    world = df.groupby('Date').sum().reset_index('Date').sort_values('Date')
    world['Country'] = 'World'
    df = pd.concat([df, eu, world])
    df = pd.merge(df, population, on=['Country'], how='left').dropna()
    df = df[['Country', 'Date', 'Code', 'Population', 'Cases', 'Deaths']]
    df = df.set_index(['Country', 'Date'])

    df['DailyCases'] = df.Cases.diff(1)
    df['DailyCases'][df['DailyCases'] < 0] = np.nan
    df['DailyMeanCases'] = df.Cases.diff(7) / 7
    df['DailyMeanCases'][df['DailyMeanCases'] < 0] = np.nan
    df['DailyRateCases'] = df.DailyMeanCases.diff(7) / 7
    df['DailyMeanCases'] = df['DailyMeanCases'].round(0)

    df['DailyDeaths'] = df.Deaths.diff(1)
    df['DailyDeaths'][df['DailyDeaths'] < 0] = np.nan
    df['DailyMeanDeaths'] = df.Deaths.diff(7) / 7
    df['DailyMeanDeaths'][df['DailyMeanDeaths'] < 0] = np.nan
    df['DailyRateDeaths'] = df.DailyMeanDeaths.diff(7) / 7
    df['DailyMeanDeaths'] = df['DailyMeanDeaths'].round(0)

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


# Convert Figure to Graph
def graph(figure, **kwargs) -> dcc.Graph:
    return dcc.Graph(figure=figure.update_layout(height=800, title_x=0.5, **kwargs))


# Plot overview with country comparisons
def plot_comparison(df: pd.DataFrame, regions: list, last_date: dt.datetime) -> {str, dhc.Div}:
    df_now = df.xs(last_date, axis=0, level=1).fillna(0).reset_index()
    df_geo = df_now[~df_now.Country.isin(['World', 'European Union'])]
    rag_scale = [(0.0, 'green'), (0.0625, 'yellow'), (0.25, 'orange'), (1.0, 'red')]
    color_palette = ['#FD3216', '#00FE35', '#6A76FC', '#FED4C4', '#FE00CE', '#0DF9FF', '#F6F926', '#FF9616',
                     '#479B55', '#EEA6FB', '#DC587D', '#D626FF', '#6E899C', '#00B5F7', '#B68E00', '#C9FBE5',
                     '#FF0092', '#22FFA7', '#E3EE9E', '#86CE00', '#BC7196', '#7E7DCD', '#FC6955', '#E48F72']

    # Plot single metric for select countries
    def plot_time_series(col: str, label: str, **kwargs) -> dcc.Graph:
        return graph(figure=df[col].unstack().transpose()[regions]
                     .figure(title=label, colors=color_palette, theme='solar', **kwargs),
                     legend_orientation='h', hovermode='x')

    # Plot current value of single metric for every country
    def plot_current(col: str, label: str, drop: list, **kwargs) -> dcc.Graph:
        ds = df_now[~df_now.Country.isin(drop)][['Country', col]].nlargest(42, columns=col).sort_values(by=col)
        return graph(figure=ds.figure(title=label, x='Country', y=col, kind='bar', orientation='h', **kwargs),
                     hovermode='y')

    # Plot Scatter of current values of two metrics for every country, optional size and/or color
    def plot_scatter(x: str, y: str, label: str, color: str = '', size: str = '',
                     drop: list = (), cutoff: int = 0) -> dcc.Graph:
        params = {'x': x, 'y': y, 'title': label,
                  'hover_name': 'Country', 'hover_data': ['Population', 'Cases', 'Deaths', 'CPM', 'DPM', 'CFR'],
                  **({'color': color, 'color_continuous_scale': rag_scale} if color else {}),
                  **({'size': size} if size else {})}
        return graph(figure=px.scatter(df_now[(~df_now.Country.isin(drop)) & (df_now.Deaths > cutoff)], **params))

    # Plot single metric for every country on a map
    def plot_geo(col: str, label: str, color_countries: bool, colors: list = ()) -> dcc.Graph:
        colors = rag_scale if color_countries else colors if colors else ['#4C33FF']
        plotter = px.choropleth if color_countries else px.scatter_geo
        params = {'title': label, 'locations': 'Code',
                  'hover_name': 'Country', 'hover_data': ['Population', 'Cases', 'Deaths', 'CPM', 'DPM', 'CFR'],
                  **({'color': col, 'color_continuous_scale': colors} if color_countries
                     else {'size': col, 'color_discrete_sequence': colors})}
        return graph(figure=plotter(df_geo, **params).update_geos(
            resolution=50, showcountries=True, countrycolor='#663399', showcoastlines=True, coastlinecolor='#663399',
            showland=True, landcolor='#E3E3E3', showocean=True, oceancolor='#ADD8E6',
            showlakes=True, lakecolor='#ADD8E6', showrivers=True, rivercolor='#ADD8E6'))

    return {
        'Scatter': dhc.Div([chart for chart in [
            plot_scatter(x='CPM', y='DPM', size='DPM', color='CFR', cutoff=1000,
                         label='Cases per Million vs Deaths per Million', ), ]]),
        'Current Cases': dhc.Div([chart for chart in [
            plot_current(col='Cases', label='Cases', drop=['World'], theme='polar', color=['#4C33FF']),
            plot_current(col='CPM', label='Cases Per Million', drop=[], theme='polar', color=['#4C33FF']), ]]),
        'Current Deaths': dhc.Div([chart for chart in [
            plot_current(col='Deaths', label='Deaths', drop=['World'], theme='polar', color=['#C70039']),
            plot_current(col='DPM', label='Deaths Per Million', drop=[], theme='polar', color=['#C70039']), ]]),
        'Maps Cases': dhc.Div([chart for chart in [
            plot_geo(col='Cases', label='Total Cases', color_countries=True),
            plot_geo(col='WeeklyCases', label='Last Week Total Cases', color_countries=True),
            plot_geo(col='CPM', label='Cases/Million', color_countries=True, colors=rag_scale),
            plot_geo(col='WeeklyCPM', label='Last Week Cases/Million', color_countries=True), ]]),
        'Maps Deaths': dhc.Div([chart for chart in [
            plot_geo(col='Deaths', label='Total Deaths', color_countries=True),
            plot_geo(col='WeeklyDeaths', label='Last Week Total Deaths', color_countries=True),
            plot_geo(col='DPM', label='Deaths/Million', color_countries=True),
            plot_geo(col='WeeklyDPM', label='Last Week Deaths/Million', color_countries=True), ]]),
        'Time-series Cases': dhc.Div([chart for chart in [
            plot_time_series(col='Cases', label='Total Cases'),
            plot_time_series(col='WeeklyCases', label='Weekly Cases (last 7 days)', kind='bar'),
            plot_time_series(col='CPM', label='Cases Per Million'),
            plot_time_series(col='WeeklyCPM', label='Weekly Cases/Million', kind='bar'), ]]),
        'Time-series Deaths': dhc.Div([chart for chart in [
            plot_time_series(col='Deaths', label='Total Deaths'),
            plot_time_series(col='WeeklyDeaths', label='Weekly Deaths (last 7 days)', kind='bar'),
            plot_time_series(col='DPM', label='Deaths Per Million'),
            plot_time_series(col='WeeklyDPM', label='Weekly Deaths/Million', kind='bar'), ]]),
        'Time-series Rates': dhc.Div([chart for chart in [
            plot_time_series(col='CFR', label='Case Fatality Rate (%)'),
            plot_time_series(col='CRR', label='7 Day Mean Reproduction Rate - Cases', logy=True),
            plot_time_series(col='DRR', label='7 Day Mean Reproduction Rate - Deaths', logy=True), ]]),
    }


# Plot regional charts
def plot_regions(df: pd.DataFrame, regions: list, last_date: dt.datetime) -> {str, dhc.Div}:
    columns, colors, titles = (list(x) for x in zip(
        ('Cases', '#4C33FF', 'Total Cases'),
        ('Deaths', '#C70039', 'Attributed Deaths'),
        ('DailyCases', '#4C33FF', 'Cases/Day'),
        ('DailyDeaths', '#C70039', 'Deaths/Day'),
        ('DailyMeanCases', '#4C33FF', '7 Day Average Cases/Day'),
        ('DailyMeanDeaths', '#C70039', '7 Day Average Deaths/Day'),
        ('DailyRateCases', '#4C33FF', 'Change 7 Day Average Cases/Day/Day'),
        ('DailyRateDeaths', '#C70039', 'Change 7 Day Average Deaths/Day/Day'),
    ))

    def plot_region(region: str) -> dcc.Graph:
        summary_values = [format_num(int(x)) for x in
                          df.loc[region].loc[last_date][['Population', 'Cases', 'Deaths', 'CPM', 'DPM', 'CFR']]]
        title = '<b>{}</b>: ' \
                '<b>{}</b> People, ' \
                '<b>{}</b> Cases, ' \
                '<b>{}</b> Deaths,<BR> ' \
                '<b>{}</b> Cases/Mil, ' \
                '<b>{}</b> Deaths/Mil, ' \
                '<b>{}%</b> Case Fatality Rate ' \
                '<i> as on {}</i><BR><BR>'.format(region, *summary_values, last_date.strftime('%d %b %Y'))
        return graph(figure=df.loc[region][columns].figure(theme='polar', title=title, subplots=True, shape=(4, 2),
                                                           legend=False, colors=colors, subplot_titles=titles),
                     hovermode='x')

    return {region: dhc.Div(plot_region(region)) for region in regions}


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
            df = transform_covid19_data(population, cases=get_covid19_data('cases'), deaths=get_covid19_data('deaths'))
            last_date = max(df.index.get_level_values(level=1))
            regions = list(df.xs(last_date, axis=0, level=1).sort_values(by='Deaths', ascending=False).index)
            short_list = regions[0:32]
            cache.update(comparison_charts := plot_comparison(df, short_list, last_date))
            cache.update(plot_regions(df, regions, last_date))
            cache.update(create_layout(keys=list(sorted(comparison_charts.keys())) + regions, shown_keys=short_list,
                                       ds=last_date.strftime('%d %b %Y')))
            log_message('Cache Updated')
        return True
    return False


# Refresh every 12th hour (43200 seconds) offset by 8 hrs (28800 - 7200 CET offset from UTC) i.e. 08:00, 20:00 CET
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
                        next_reload_data_at = ((1 + (int(time.time()) // 43200)) * 43200) + 21600
                    while (wait := next_reload_data_at - int(time.time())) > 0:
                        log_message('next reload ' + dt.datetime.fromtimestamp(next_reload_data_at).strftime('%H:%M'))
                        time.sleep(wait / 2)

    if not cache_loop_lock.locked():
        threading.Thread(target=loop_update_cache, daemon=True).start()


@server.route(status_link)
def status():
    return 'updating cache' if cache_update_lock.locked() \
        else 'serving {} items cached at {}'.format(len(cache), cache['report_date'])


def text_box(lines: [str, ...]) -> str:
    rows = len(lines)
    cols = max(len(line) for line in lines)
    return '{0}{1} rows={4} cols={5} {2}{3}{0}/{1}{2}'.format('<', 'textarea', '>', '\n'.join(lines), rows, cols)


@server.route(code_link)
def code():
    page = '<html><head></head><body>{}</body></html>'
    return page.format(text_box(lines=inspect.getsource(sys.modules[__name__]).splitlines()))


@server.route(ld_link)
def logs():
    page = '<html><head><meta http-equiv="refresh" content="10"></head><body>{}</body></html>'
    if ld_file:
        with open(ld_file) as infile:
            return page.format(text_box([line.strip() for line in infile.readlines()]))
    return page.format('')


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
        ld_file = os.path.abspath(__file__)[:-3] + ('-d.log' if args.dev else '.log')
        cache = create_layout()
        if args.dev:
            __import__('requests_cache').install_cache('cache', expire_after=12 * 3600)
        __import__('atexit').register(log_message, 'http://{}:{} down'.format(host, port))
        log_message('http://{}:{} up'.format(host, port))
        app.run_server(host=host, port=port)
