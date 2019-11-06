# downloading nyc zipcode date and moving to warehouse
# !python --version

import os
import sys
import csv
import requests
import urllib.request
import pandas as pd
import psycopg2

sys.path.append('../')
from envir import config
import imp
imp.reload(config)

if os.path.exists(config.shared+'ny_zip_demographics.csv') == False:
    zip_url = 'https://data.ny.gov/api/views/juva-r6g2/rows.csv?accessType=DOWNLOAD'
    urllib.request.urlretrieve(zip_url, config.shared+'ny_zip_demographics.csv')

zipcodes = pd.read_csv(config.shared+'ny_zip_demographics.csv')

zipcodes.columns = zipcodes.columns.str.replace(' ', '_').str.lower()

zipcodes.drop(['file_date'], axis=1, inplace=True)

# connect to warehouse
con = psycopg2.connect(config.con)
cur = con.cursor()
cur.execute('''create table if not exists stag.ny_zipcodes(
                county_name varchar, 
                state_fips int, 
                county_code int,
                county_fips int, 
                zip_code varchar
                );''')                                      
con.commit()

zipcodes.to_csv(config.shared+'temp.csv', index=False)

con = psycopg2.connect(config.con)
cur = con.cursor()

with open(config.shared+'temp.csv', 'r') as f:
    # Notice that we don't need the `csv` module.
    next(f) # Skip the header row.
    cur.copy_from(f, 'stag.ny_zipcodes', sep=',')
con.commit()

# pd.read_sql('select * from stag.ny_zipcodes', con)