#!/usr/bin/env python3
# get_ipython().system('python --version')

import os
import sys
import datetime
import pandas as pd
import geopandas as gpd
import tweepy
import psycopg2
from sqlalchemy import create_engine
from datetime import date, timedelta

import config


# ### Read in the Downloaded Files
us_zips = pd.read_csv(config.shared+'2019_Gaz_zcta_national.txt', sep="\t")
ny_zips = pd.read_csv(config.shared+'ny_zip_demographics.csv', sep=',')
nyc_zips = ny_zips[ny_zips['County Name'].isin(['New York', 'Kings', 'Queens','Richmond','Bronx'])]
nyc_zips = pd.merge(nyc_zips, us_zips, how='inner', left_on='ZIP Code', right_on = 'GEOID')
nyc_zips.rename(str.strip, axis='columns', inplace=True)

# then read it in with geopandas, reading in the shape file with the function 
pumas = gpd.GeoDataFrame.from_file(config.shared+'/NYC_PUMAS' +'/'+ 'NYC_PUMAS.shp')
pumas['centroid'] = pumas['geometry'].centroid
pumas['lon'] = pumas['centroid'].apply(lambda p: p.x)
pumas['lat'] = pumas['centroid'].apply(lambda p: p.y)

# then read it in with geopandas, reading in the shape file with the function 
census = gpd.GeoDataFrame.from_file(config.shared+'/NYC_CENSUS' +'/'+ 'NYC_CENSUS.shp')
census['centroid'] = census['geometry'].centroid
census['lon'] = census['centroid'].apply(lambda p: p.x)
census['lat'] = census['centroid'].apply(lambda p: p.y)

# ### Set Connection to DB
engine = create_engine('postgresql+psycopg2:///yelp_abcs')
conn = engine.connect()

# ### Search for Tweets By Census Tract and Search Terms
auth = tweepy.OAuthHandler(config.consumer_key, config.consumer_secret)
auth.set_access_token(config.access_token, config.access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

delivery_apps=['doordash', 'ubereats', 'postmates', 'grubhub', 'seamless', 'caviar']
key_words= ['snack', 'pizza', 'burger', 'hungry','', 'food', 'drinks', 'late night', 'delivery', 'order', 'takeout', 'dinner', 'late night eats']

fr_date = date.today() - timedelta(7) # twitter api only allows upto 7 days on the basic service

for index, row in pumas.iterrows():
    for app in delivery_apps:
        for word in key_words:
            print(app, row['puma'], word)
            query = word + ' ' + app + ' -filter:retweets'
            location = str(row['lat']) + ',' + str(row['lon']) + ',2km'
            tweets = tweepy.Cursor(api.search, q=query, geocode=location, lang='en',
                                   result_type = 'recent', since=fr_date).items()
            df = pd.DataFrame(columns = ['created_at', 'user_name', 'user_location', 'id', 'text',
                                         'place', 'coordinates', #twitter returns
                                         'keyword_search_app', 'puma'])
            for t in tweets:
                df = df.append({'created_at': str(t.created_at),
                                'user_name': str(t.user.name),
                                'user_location': str(t.user.location),
                                'id': str(t.id),
                                'text': str(t.text),
                                'place': str(t.place), 
                                'coordinates': str(t.coordinates),
                                'keyword_search_app': str(app),
                                'puma': str(row['puma'])},
                               ignore_index = True)
            df.to_csv(config.shared+'twitter_results/'+app+'_'+row['puma']+'_'+word+'.csv')
            df.to_sql('twitter_results', engine, schema = 'stag', if_exists='append', index=False)