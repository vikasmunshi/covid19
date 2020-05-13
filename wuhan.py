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


# retrieve data from URL and return Pandas DataFrame
def data_frame_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(requests.get(url).content.decode()))


# Population Data
def get_population() -> pd.DataFrame:
    url = 'http://api.worldbank.org/countries/all/indicators/SP.POP.TOTL?format=csv'
    population = data_frame_from_url(url)[['Country Name', 'Country Code', '2018']]
    population.columns = ['Country', 'Code', 'Population']
    for country, name_change, missing_data in [
        ['Bahamas, The', 'Bahamas', None], ['Brunei Darussalam', 'Brunei', None],
        ['Congo, Dem. Rep.', 'Congo (Kinshasa)', None], ['Congo, Rep.', 'Congo (Brazzaville)', None],
        ['Czech Republic', 'Czechia', None], ['Egypt, Arab Rep.', 'Egypt', None], ['Gambia, The', 'Gambia', None],
        ['Iran, Islamic Rep.', 'Iran', None], ['Korea, Rep.', 'Korea, South', None],
        ['Kyrgyz Republic', 'Kyrgyzstan', None], ['Lao PDR', 'Laos', None], ['Myanmar', 'Burma', None],
        ['Russian Federation', 'Russia', None], ['Slovak Republic', 'Slovakia', None],
        ['Syrian Arab Republic', 'Syria', None], ['United States', 'US', None], ['Venezuela, RB', 'Venezuela', None],
        ['Yemen, Rep.', 'Yemen', None], ['Taiwan*', None, ['TWN', 23780452]]
    ]:
        if name_change:
            population['Country'][population.Country == country] = name_change
        if missing_data:
            population.loc[len(population)] = [country] + missing_data
    return population


# Retrieve Covid-19 Data and plot charts
def get_covid19_data(label: str) -> pd.DataFrame:
    urls = {
        'cases': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv',
        'deaths': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv',
        'recovered': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv',
    }
    df = data_frame_from_url(urls[label]).drop(0)[['Country/Region', 'Province/State', 'Date', 'Value']]
    df.Date = pd.to_datetime(df.Date)
    df.Value = df.Value.astype(int)
    df = df.groupby(['Country/Region', 'Date']).sum().reset_index()
    df.columns = ['Country', 'Date', label.title()]
    return df.sort_values(by=['Country', 'Date'])


# Retrieve Covid-19 Data and plot charts
def plot_covid19_data(population: pd.DataFrame) -> {str: html.Div}:
    df = get_covid19_data('cases')
    df = pd.merge(df, get_covid19_data('deaths'), on=['Country', 'Date'], how='inner')
    df = pd.concat([df, df.groupby('Date').sum().reset_index('Date').sort_values('Date')]).fillna('World')
    df = pd.merge(df, population, on=['Country'], how='left').dropna()
    df = df[['Country', 'Date', 'Code', 'Population', 'Cases', 'Deaths']]
    df = df.set_index(['Country', 'Date'])
    df['CFR'] = df.Deaths / df.Cases
    df['CPM'] = 10 ** 6 * df.Cases / df.Population
    df['DPM'] = 10 ** 6 * df.Deaths / df.Population
    df['WeeklyCases'] = df.Cases.diff(7)
    df['WeeklyDeaths'] = df.Deaths.diff(7)
    df['WeeklyCPM'] = df.CPM.diff(7)
    df['WeeklyDPM'] = df.DPM.diff(7)
    df['CRR'] = ((df.WeeklyCases / df.WeeklyCases.shift(7)) ** (1 / 7)).replace(np.inf, np.nan)

    def clean_dataseries(ds, z: int = 6) -> pd.Series:
        ds[ds < 0] = np.nan
        x = ds.unstack().transpose()
        x = x.mask((x - x.mean()).abs() > z * x.std())
        return x.transpose().stack()

    df.CFR = clean_dataseries(df.CFR, z=2)
    df.CRR = clean_dataseries(df.CRR, z=3)
    df.WeeklyCases = clean_dataseries(df.WeeklyCases)
    df.WeeklyDeaths = clean_dataseries(df.WeeklyDeaths)
    df.WeeklyCPM = clean_dataseries(df.WeeklyCPM)
    df.WeeklyDPM = clean_dataseries(df.WeeklyDPM)

    last_date = max(df.index.get_level_values(level=1))
    regions_sorted_by_deaths = list(df.xs(last_date, axis=0, level=1).sort_values(by='Deaths', ascending=False).index)
    countries_to_show_in_overview = regions_sorted_by_deaths[0:31] + ['Taiwan*']

    def plot_dataseries(ds: pd.Series, label: str, **kwargs):
        return ds.unstack().transpose()[countries_to_show_in_overview].figure(title=label, theme='polar', **kwargs)

    charts = {'Overview': html.Div([
        dcc.Graph(figure=chart.update_layout(height=800, hovermode='x', title_x=0.5))
        for chart in [
            plot_dataseries(df.WeeklyCases, 'Weekly Cases (last 7 days)', kind='bar'),
            plot_dataseries(df.Cases, 'Total Cases'),
            plot_dataseries(df.CPM, 'Cases Per Million'),
            plot_dataseries(df.WeeklyCPM, 'Weekly Cases Per Million (last 7 days)'),

            plot_dataseries(df.WeeklyDeaths, 'Weekly Deaths (last 7 days)', kind='bar'),
            plot_dataseries(df.Deaths, 'Total Deaths'),
            plot_dataseries(df.DPM, 'Deaths Per Million'),
            plot_dataseries(df.WeeklyDPM, 'Weekly Deaths Per Million (last 7 days)'),

            plot_dataseries(df.CFR, 'Case Fatality Rate').update_layout(yaxis={'tickformat': ',.0%'}),
            plot_dataseries(df.CRR, 'Case Reproduction Rate (last 7 day average)', logy=True), ]])}

    df = df[['Cases', 'WeeklyCases', 'Deaths', 'WeeklyDeaths', 'CFR', 'CRR']]
    df.columns = ['Cases', 'Weekly Cases (last 7 days)', 'Deaths', 'Weekly Deaths (last 7 days)',
                  'Case Fatality Rate', 'Case Reproduction Rate']
    regional_population = {row['Country']: int(row['Population']) for row in population.dropna().to_dict('records')}

    for region in regions_sorted_by_deaths:
        charts[region] = html.Div(dcc.Graph(figure=df.loc[region].figure(
            title='{} ({:,} million people)'.format(region, regional_population[region] // 10 ** 6),
            theme='polar', subplots=True, shape=(3, 2), shared_xaxes=True, subplot_titles=True,
            colors=['#000000'], legend=False).update_layout(height=780, hovermode='x', title_x=0.5)))

    title = 'Wuhan Corona Virus Pandemic Stats {} (retrieved {})'.format(last_date.strftime('%d %b %Y'),
                                                                         datetime.now().strftime('%d %b %Y %H:%M'))
    charts['layout'] = layout(title=title, keys=['Overview'] + regions_sorted_by_deaths,
                              show_keys=countries_to_show_in_overview)
    return charts


def layout(title: str, keys: list, show_keys: list = ()) -> html.Div:
    return html.Div([
        html.Div(html.H6(title)),
        dcc.Dropdown(id='region', options=[{'label': k, 'value': k} for k in keys],
                     value=keys[0] if not show_keys else show_keys, multi=bool(show_keys)),
        html.Div(id='page-content')])


# Cache Charts
cache = {'Loading...': html.Div(['Retrieving Data ... ', html.A('refresh', href='/')]),
         'layout': layout(title='Wuhan Corona Virus Pandemic Stats', keys=['Loading...'])}


# Refresh every 12th hour i.e. 00:00 UTC, 12:00 UTC (43200 seconds)
def update_cache(population: pd.DataFrame):
    while True:
        try:
            for k, v in plot_covid19_data(population).items():
                cache[k] = v
            print(datetime.now(), 'Cache updated', flush=True)
        except Exception as e:
            print(datetime.now(), 'Exception occurred while updating cache\n', str(e), flush=True)
            update_at = (1 + int(time()) // 3600) * 3600
        else:
            update_at = (1 + int(time()) // 43200) * 43200
        while (diff := update_at - int(time())) > 0:
            print(datetime.now(), 'Next cache update in {} seconds'.format(diff), flush=True)
            sleep(diff / 10)


# Cufflinks
cf.go_offline()
# Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Wuhan Corona Virus Pandemic Stats'
app.layout = lambda: cache['layout']


@app.callback(dash.dependencies.Output('page-content', 'children'), [dash.dependencies.Input('region', 'value')])
def update_output(value):
    return [cache[v] for v in value] if isinstance(value, list) else cache[value]


if __name__ == '__main__':
    Thread(target=update_cache, kwargs={'population': get_population()}, daemon=True).start()
    app.run_server(host='0.0.0.0')
