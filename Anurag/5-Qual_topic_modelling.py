import os
import json
import string
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import wordcloud
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
import logging
import gensim
from gensim import corpora
import spacy


def preprocess(tokens):
    tokens_nop = [t for t in tokens if t not in string.punctuation]
    tokens_nop = [t.lower() for t in tokens_nop]
    # wnl = nltk.WordNetLemmatizer()
    stop = stopwords.words('english')
    stop.extend(['armenian','armenia', 'job', 'title', 'position', 'location', 'responsibilities', 'application', 'procedures',
                 'deadline', 'required', 'qualifications', 'renumeration', 'salary', 'date', 'company'])
    tokens_nostop = [t for t in tokens_nop if t not in stop]
    # tokens_lem = [wnl.lemmatize(t) for t in tokens_nostop]
    tokens_clean = [t for t in tokens_nostop if len(t) >= 3]  # simple way to remove the offending " punctuations
    return tokens_clean



def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]


def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]


# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


df_ori = pd.read_csv('../data/data job posts.csv')
df_ori.head()
print(df_ori.shape)
df = df_ori.drop_duplicates(['RequiredQual'])
print(df.shape)
print("Removed {0} duplicates (based on RequiredQual)".format(df_ori.shape[0] - df.shape[0]))

print(df['RequiredQual'].head())

df['RequiredQual'] = df['RequiredQual'].astype(str)
df['RequiredQual_tokens'] = df.RequiredQual.map(word_tokenize)
# df['RequiredQual_tokens'] = df.RequiredQual_tokens.apply(set)
df['RequiredQual_processed'] = df.RequiredQual_tokens.apply(preprocess)
# df['jobpost_processedtext'] = df.jobpost_processed.apply(lambda x: ' '.join(x))

# Build the bigram and trigram models
bigram = gensim.models.Phrases(df['RequiredQual_processed'], min_count=5, threshold=100)  # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[df['RequiredQual_processed']], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

## no clue what this does?
print(trigram_mod[bigram_mod[df.RequiredQual_tokens[5]]])

df.RequiredQual_processed = make_bigrams(df.RequiredQual_processed)
df.RequiredQual_processed = lemmatization(df.RequiredQual_processed, allowed_postags=['NOUN', 'VERB'])  # 'ADJ',, 'ADV'])

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
dictionary = corpora.Dictionary(df['RequiredQual_processed'])
print(dictionary)
dictionary.filter_extremes(no_below=3, no_above=0.7)
print(dictionary)

topic_num = 4

# Use the dictionary to prepare a DTM (using TF)
dtm_train = [dictionary.doc2bow(d) for d in df['RequiredQual_processed']]
lda = gensim.models.ldamodel.LdaModel(dtm_train, num_topics=topic_num, alpha='auto', chunksize=30, id2word=dictionary,
                                      passes=20, random_state=432)
lda.show_topics()
print(lda.show_topics(num_words=20))

dtopics_train = lda.get_document_topics(dtm_train)
# print topic distribution for 1st 5 rows
for i in range(0, 5):
    print(dtopics_train[i])

# get and plot the distribution of the topics
from operator import itemgetter
import matplotlib.style as style
style.use('ggplot')
style.use('seaborn-poster')
fig, ax = plt.subplots(figsize=(16, 9))
top_train = [max(t, key=itemgetter(1))[0] for t in dtopics_train]
plt.hist(top_train, bins=topic_num-1)
plt.title('Topic Frequencies')
plt.show()
