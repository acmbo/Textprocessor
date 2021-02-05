import sys
import os
import wikipedia as wk
import json

#change sys folder
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

# open premade citynames
f = open("citynames.txt", "r")

txt = f.read()
txt = txt.replace('\n','')
txt_arr = txt.split(';')


# scrape wikipedia for an Exampletexts
dictcity={}

for city in txt_arr:
    if city != '':
        page = wk.page(city)
        dictcity[page.title] = page.content.replace('\n',' ')


with open('example_data.txt', 'w') as outfile:
    json.dump(dictcity, outfile)
