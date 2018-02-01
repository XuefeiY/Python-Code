# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:33:46 2018

"""

# =============================================================================
# Import packages
# =============================================================================
import re
import pandas as pd
from user_agents import parse



# =============================================================================
# Helper functions
# =============================================================================
# returns a list of all indices of char inside string
def find_chars(string, char):
    return [i for i, ltr in enumerate(string) if ltr == char]

# only keep rows whose uri_query has a website id in it
def indicator_site(row):
    return re.search('websiteConfigId=', row['uri_query']) is not None

# scape webiste id from uri_query
def extract_site_id(row):
    return re.search('websiteConfigId=(\d+)*', row['uri_query']).group(1)

# parse user agent into string
def ua_parse(row):
    user_agent = parse(row['user_agent'])
    return str(user_agent)

# get the device of user
def ua_device(row):
    user_agent = parse(row['user_agent'])
    if user_agent.is_mobile:
        val = 'mobile'
    elif user_agent.is_pc:
        val = 'pc'
    elif user_agent.is_tablet:
        val = 'tablet'
    elif user_agent.is_bot:
        val = 'bot'
    else:
        val = 'other'
    return val


# =============================================================================
# Main function
# =============================================================================
def log_parse(app, file, date):
    # open file
    path = "blahblah"+app+"_u_ex"+date+"_adv.log"
    infile = path + "\\" + file
    with open(infile, 'r') as f:
        f = f.readlines()
    
    # scrape useful info and write it into a dataframe
    lines = [line for line in f if len(line) > 200]   # pass lines without user agent info
    loglist = []
    for s in lines:
        line = s.strip()
        tmp = line.split(" ")
        uri_stem = tmp[6]
        uri_query = tmp[7]
        doublequotes = find_chars(line, '"')
        useragent_start = doublequotes[4]+1
        useragent_end = doublequotes[5]
        useragent = line[useragent_start:useragent_end]
        loglist.append({"uri_stem": uri_stem, "uri_query": uri_query, "user_agent": useragent})
    df = pd.DataFrame.from_dict(loglist)
    
    # website
    df['ind_site']  = df.apply(indicator_site, axis=1)
    df = df[df['ind_site']]
    df['Website ID']  = df.apply(extract_site_id, axis=1)
    df_reduced = df
    del df_reduced['ind_site']
    
    # user agent
    df_reduced['user_agent_string'] = df_reduced.apply(ua_parse, axis=1)
    df_reduced['user_agent_device'] = df_reduced.apply(ua_device, axis=1)
    
    # write into csv
    df_reduced['application'] = app
    df_reduced['date'] = date
    cols = ['application', 'date', 'uri_query', 'user_agent', 'user_agent_string', 'Website ID', 'uri_stem', 'Website Name', 'user_agent_device']
    df_final = df_reduced[cols]
    df_final.to_csv(date+"_mpv_log_"+app+".csv", index = False)
 
    



    

        
