import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import pandas as pd

import acquire


def basic_clean():
    original = acquire.make_new_request()
    article = original.copy()
    article['body'] = article.body.str.lower()
    article['title'] = article.title.str.lower()
    return article
article = basic_clean()


