#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
""" Visualize Wuhan Corona Virus Stats """
from io import StringIO
from os import path
from sys import argv

import matplotlib.pyplot as plt
import pandas as pd
import requests
import requests_cache
from matplotlib import rc
from matplotlib import style
from matplotlib import ticker

URLS = {
    'population': 'https://datahub.io/JohnSnowLabs/population-figures-by-country/r/population-figures-by-country-csv.csv',
    'confirmed cases': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_confirmed_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_confirmed_global.csv',
    'deaths': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_deaths_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_deaths_global.csv',
    'recovered': 'https://data.humdata.org/hxlproxy/data/download/time_series_covid19_recovered_global_narrow.csv?dest=data_edit&filter01=explode&explode-header-att01=date&explode-value-att01=value&filter02=rename&rename-oldtag02=%23affected%2Bdate&rename-newtag02=%23date&rename-header02=Date&filter03=rename&rename-oldtag03=%23affected%2Bvalue&rename-newtag03=%23affected%2Binfected%2Bvalue%2Bnum&rename-header03=Value&filter04=clean&clean-date-tags04=%23date&filter05=sort&sort-tags05=%23date&sort-reverse05=on&filter06=sort&sort-tags06=%23country%2Bname%2C%23adm1%2Bname&tagger-match-all=on&tagger-default-tag=%23affected%2Blabel&tagger-01-header=province%2Fstate&tagger-01-tag=%23adm1%2Bname&tagger-02-header=country%2Fregion&tagger-02-tag=%23country%2Bname&tagger-03-header=lat&tagger-03-tag=%23geo%2Blat&tagger-04-header=long&tagger-04-tag=%23geo%2Blon&header-row=1&url=https%3A%2F%2Fraw.githubusercontent.com%2FCSSEGISandData%2FCOVID-19%2Fmaster%2Fcsse_covid_19_data%2Fcsse_covid_19_time_series%2Ftime_series_covid19_recovered_global.csv',
}

requests_cache.install_cache(path.split(path.split(__file__)[0])[1], expire_after=12 * 3600)


def data_frame_from_url(url_name: str) -> pd.DataFrame:
    return pd.read_csv(StringIO(requests.get(URLS[url_name]).content.decode()))


population = data_frame_from_url('population').set_index('Country').iloc[:, [-1]].dropna(0)
population.columns = ['Population']
population.Population = population.Population.astype(int)
for country, name_change, missing_data in [
    ('Bahamas, The', 'Bahamas', None),
    ('Brunei Darussalam', 'Brunei', None),
    ('Congo, Dem. Rep.', 'Congo (Kinshasa)', None),
    ('Congo, Rep.', 'Congo (Brazzaville)', None),
    ('Czech Republic', 'Czechia', None),
    ('Egypt, Arab Rep.', 'Egypt', None),
    ('Gambia, The', 'Gambia', None),
    ('Iran, Islamic Rep.', 'Iran', None),
    ('Korea, Rep.', 'Korea, South', None),
    ('Kyrgyz Republic', 'Kyrgyzstan', None),
    ('Lao PDR', 'Laos', None),
    ('Macedonia, FYR', 'North Macedonia', None),
    ('Myanmar', 'Burma', None),
    ('Russian Federation', 'Russia', None),
    ('Slovak Republic', 'Slovakia', None),
    ('Syrian Arab Republic', 'Syria', None),
    ('Taiwan*', None, 23780452),
    ('United States', 'US', None),
    ('Venezuela, RB', 'Venezuela', None),
    ('Yemen, Rep.', 'Yemen', None)
]:
    if name_change:
        population.loc[name_change] = population.loc[country]
    if missing_data:
        population.loc[country] = missing_data
population = population / 10 ** 6
population.loc['Rest'] = population.loc['World'] - population.loc['China']

data_frames = {}
for name, data_frame in ((k, data_frame_from_url(k)) for k in URLS.keys() if k != 'population'):
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
    data_frame = data_frame.resample('7D').last()
    data_frames[name] = data_frame
    data_frames[name + ' per million'] = data_frame.div(population.Population, axis=1).dropna(axis=1)
    data_frames['increment per million in ' + name] = data_frames[name + ' per million'].diff().dropna().astype(int)
data_frames['mortality rate'] = (data_frames['deaths'] / data_frames['confirmed cases']).fillna(0)

countries = list(argv[1:]) or list(data_frames['mortality rate'].columns)
style.use('fivethirtyeight')
rc('font', size=9)
rc('axes', titlesize=9)
rc('axes', labelsize=9)
fig, axes = plt.subplots(nrows=2, ncols=3, sharex='all')
axes_selection = (
    ('confirmed cases', axes[0, 0], True,),
    ('confirmed cases per million', axes[0, 1], True,),
    ('increment per million in confirmed cases', axes[0, 2], True,),
    ('deaths', axes[1, 0], True,),
    ('deaths per million', axes[1, 1], True,),
    ('mortality rate', axes[1, 2], False,),
)
for label, ax, show_legend in axes_selection:
    df = data_frames[label][countries]
    df.plot(ax=ax, title=label, grid=True, legend=show_legend, kind='bar', rot=0)
    ax.axes.get_xaxis().get_label().set_visible(False)
    ax.axes.get_yaxis().get_label().set_visible(False)
    tick_labels = [''] * len(df.index)
    tick_labels[::4] = [item.strftime('%d %b') for item in df.index[::4]]
    tick_labels[::26] = [item.strftime('%d %b\n%Y') for item in df.index[::26]]
    ax.xaxis.set_major_formatter(ticker.FixedFormatter(tick_labels))

plt.get_current_fig_manager().set_window_title('Wuhan Corona Virus Pandemic')
plt.show()
plt.close(fig=fig)
