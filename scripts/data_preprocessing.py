import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import string

data = pd.read_parquet('../data/raw/2010_de.parquet')

model_data = data[data['Index'] == 'D']


def normalize(text):
    return text.lower(data)


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_numbers(text):
    return re.sub(r'\d+', '', text)


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stop_words])


def clean_data(model_data):
    model_data['cleaned_data'] = model_data['ET and AB'].apply(normalize)
    model_data['cleaned_data'] = model_data['cleaned_data'].apply(remove_punctuation)
    model_data['cleaned_data'] = model_data['cleaned_data'].apply(remove_numbers)
    model_data['cleaned_data'] = model_data['cleaned_data'].apply(remove_stopwords)
    return model_data
