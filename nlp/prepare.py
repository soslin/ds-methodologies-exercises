import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import acquire


def handle_white_space(s):
    s = s.strip()
    s = s.replace('\n', ' ')
    return re.sub(r"[^a-z0-9'\s]", '', s)

def unicode_normalize(s): 
    return unicodedata.normalize('NFKD', s)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')

def prepare_article():
    original = acquire.make_new_request()
    article = original.copy()
    article['body'] = article.body.str.lower()
    article['body'] = article.body.apply(handle_white_space)
    article['body'] = article.body.apply(unicode_normalize)
    return article
article = prepare_article()


