from requests import get
import pandas as pd
from bs4 import BeautifulSoup
import os


def get_blog_posts():
    filename = './codeup_blog_posts.csv'
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return make_new_request()
    
    
def make_dictionary_from_article(url):
    #set header and user agent to increase likelihood that you request get the response you want
    headers = {'user-agent' : 'Codeup Bayes Student Example'}
    #This is the actuall HTTP GET request that python will send across the internet
    response = get(url, headers = headers)
    #response.text is a single string of all the html from that page
    soup = BeautifulSoup(response.content)
    
    title = soup.title.get_text()
    body = soup.select('div.mk-single-content.clearfix')[0].get_text()
    
    return {
        'title': title,
        'body' : body
    }

def make_new_request():
    urls = [
        "https://codeup.com/codeups-data-science-career-accelerator-is-here/",
        "https://codeup.com/data-science-myths/",
        "https://codeup.com/data-science-vs-data-analytics-whats-the-difference/",
        "https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/",
        "https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/",
    ]
    output = []
    
    for url in urls:
        output.append(make_dictionary_from_article(url))

    df = pd.DataFrame(output)
    df.to_csv('codeup_blog_posts.csv')
        
    return df

df = make_new_request()