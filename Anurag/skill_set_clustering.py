import os
import json
import string
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import pandas as pd

df_ori = pd.read_csv('../data/data job posts.csv')
df_ori.head()
print(df_ori.shape)
df = df_ori.drop_duplicates(['RequiredQual'])
print(df.shape)
print("Removed {0} duplicates (based on RequiredQual)".format(df_ori.shape[0]-df.shape[0]))

df["RequiredQual"].head()

df["RequiredQual"] = df["RequiredQual"].astype(str)

df['RequiredQual_token'] = df['RequiredQual'].map(word_tokenize)

## Preprocess the tokens using the following steps:
### a. Remove punctuation
### b. Change to lower case
### c. remove stop words
### d. Lemmatize nouns only refer [here](https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word)
### e. Keep only the tokens that are of length 3 or more

def preprocess(tokens):
    tokens_nop = [t for t in tokens if t not in string.punctuation]
    tokens_nop = [t.lower() for t in tokens_nop]
    wnl = nltk.WordNetLemmatizer()
    stop = stopwords.words('english')
    tokens_nostop = [t for t in tokens_nop if t not in stop]
    tokens_lem = [wnl.lemmatize(t) for t in tokens_nostop]
    tokens_clean = [t for t in tokens_lem if len(t) >= 3]
    return tokens_clean

df['RequiredQual_processed'] = df.RequiredQual_token.apply(preprocess)

from collections import defaultdict

# set the default value when a key is not present in the default dict
# sparse matrix is a dict of dicts, the value is the co-occurance value
sparse_matrix = defaultdict(lambda: defaultdict(lambda: 0))

for job_qualification_tokens in df['RequiredQual_processed']:
    for word1 in job_qualification_tokens:
        for word2 in job_qualification_tokens:
            sparse_matrix[word1][word2]+=1
            
print(sparse_matrix)