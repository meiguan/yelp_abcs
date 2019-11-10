#!/usr/bin/env python
# coding: utf-8

# ### Downloading Basic Yelp Restaurant Data - script development

import os
import sys
import datetime
import requests
import urllib.request
import json
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

sys.path.append('../')
from envir import config

local = 'New York, Manhattan, Greenwich Village'

if os.path.exists(config.shared+'yelp_categories.json') == False: 
    # download json of categories
    cat_url = 'https://www.yelp.com/developers/documentation/v3/all_category_list/categories.json'
    urllib.request.urlretrieve(cat_url, config.shared+'yelp_categories.json')

with open(config.shared+'yelp_categories.json') as f:
    data = json.load(f)

cuisines = []
for i in range(len(data)):
    if data[i]['parents']==['restaurants']:
        cuisines.append(data[i]['title'])

totals_df = pd.DataFrame(columns=['location', 'cuisine', 'total_businesses'])

# conversion function:
def dict2json(dictionary):
    return json.dumps(dictionary, ensure_ascii=False)

for food in cuisines:
    print(food)
    offset = 0
    limit = 50
    total = 1000
            
    res_df = pd.DataFrame()
        
    while offset < total and (offset+limit)<1000:
        # define the api call
        url='https://api.yelp.com/v3/businesses/search'
        headers = {'Authorization': 'Bearer %s' % config.yelpApi}
        params = {'categories':'restaurants', 'term':food, 'location':local, 'limit':limit, 'offset':offset}
            
        # Making an initial request to the API and saving the first set of results
        req = requests.get(url, params=params, headers=headers)
        res_json = json.loads(req.text)
            
        if 'error' in res_json:
            sys.exit()
        else:
            try:
                res_df = res_df.append(pd.DataFrame.from_dict(res_json['businesses']))
                # current placement
                sys.stdout.write('Currently {} at {} out of {}.'.format(local, params['offset'], res_json['total']))
                offset = offset+50+1
                total = res_json['total']
            except:
                continue
    try:
        if res_json['total'] > 0:
            totals_df = totals_df.append([{'location': local,
                                            'cuisine':food,
                                            'total_businesses':res_json['total']}], ignore_index=True)
            # overwrite the dict column with json-strings
            res_df['categories'] = res_df.categories.map(dict2json)
            res_df['coordinates'] = res_df.coordinates.map(dict2json)
            res_df['location'] = res_df.location.map(dict2json)
            res_df['cuisine'] = food
            res_df['search_location'] = local
            res_df['api_access_ts'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            engine = create_engine('postgresql+psycopg2:///yelp_abcs')
            con = engine.connect()
            res_df.to_sql('yelp_businesses', engine, schema = 'stag', if_exists='append', index=False)
    except:
        continue
