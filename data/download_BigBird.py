#!/usr/bin/env python
from bs4 import BeautifulSoup
from argparse import ArgumentParser
import requests
import os
import shutil


parser = ArgumentParser()
parser.add_argument('--data_type', type=str, help="rgbd.tgz, rgb_highres.tgz or processed.tgz")
parser.add_argument('--dir', type=str, help='directory to download data to')

args = parser.parse_args()
data_type = args.data_type
path = args.dir

link  = "http://rll.berkeley.edu/bigbird/aliases/8b4888c959/"
html = requests.get(link)
soup = BeautifulSoup(html.text)

links = soup.find_all('a')


for l in links:
    url = l.get('href')
    if data_type in url:
        object_name = url.split('/')[-2]
        print "Downloading %s..." % (object_name)
        if not os.path.isdir(path+object_name):
            os.makedirs(path+object_name)
        r = requests.get(link+url, stream=True)
        with open(path+object_name+'/rgbd.tgz', 'wb') as f:
            shutil.copyfileobj(r.raw, f)


print "Download finished!"
