# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:45:35 2017

@author: xuefei.yang
"""

#import the library used to query a website
import urllib.request  as urllib2 

#specify the url
wiki = "https://en.wikipedia.org/wiki/List_of_Olympic_Games_host_cities"

#Query the website and return the html to the variable 'page'
page = urllib2.urlopen(wiki)

#import the Beautiful soup functions to parse the data returned from the website
from bs4 import BeautifulSoup

#Parse the html in the 'page' variable, and store it in Beautiful Soup format
soup = BeautifulSoup(page)

print (soup.prettify())


soup.title
soup.title.string
soup.a 

all_links=soup.find_all("a")
for link in all_links:
    print (link.get('href'))
    
all_tables=soup.find_all('table')
right_table=soup.find('table', class_='wikitable sortable')
right_table


#Generate lists
A=[]
B=[]
C=[]
D=[]
E=[]
F=[]
G=[]
H=[]
for row in right_table.findAll("tr"):
    cells = row.findAll('td')
    if len(cells)==8: #Only extract table body not heading
        A.append(cells[0].find(text=True))
        B.append(cells[1].find(text=True))
        C.append(cells[2].find(text=True))
        D.append(cells[3].find(text=True))
        E.append(cells[4].find(text=True))
        F.append(cells[5].find(text=True))
        G.append(cells[6].find(text=True))
        
        
https://www.analyticsvidhya.com/blog/2015/10/beginner-guide-web-scraping-beautiful-soup-python/
