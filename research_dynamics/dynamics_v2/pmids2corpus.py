import pymysql
import pickle
import json

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['svg.fonttype'] = 'none'

import time
import hdbscan

import spacy
from gensim.models import Word2Vec

import random
import re
import unicodedata


# note this is very similar to pmids2vec - might be good to refactor 


def pmids2corpus(PMIDs, save_path):
    
    # database parameters
    db_name = 'test_pubmed'
    config_path = '/home/brendan/Projects/AttentionWildfires/attention_wildfires/mysql_config.json'

    with open(config_path, 'r') as f:
        config_data = json.load(f)

    client_config = {'database': db_name,
                    'user': config_data['user'],
                     'password': config_data['lock']}

    db = pymysql.connect(**client_config)
    
    # get the title & abstract text associated with pmids
    str_fmt = ', '.join([str(pmid) for pmid in PMIDs])

    sql = '''SELECT A.title, A.abstract
            FROM abstracts as A
            WHERE A.pmid IN ({})'''.format(str_fmt)

    start_time = time.time()
    cursor = db.cursor()
    cursor.execute(sql)
    end_time = time.time()
    elapsed = end_time - start_time
    print("SQL join executed in {} s".format(elapsed))

    start_time = time.time()
    titles = []
    abstracts = []
    for i,row in enumerate(cursor):
        print_block_len = 100000
        if (i+1) % print_block_len == 0:
            print('fetched {} rows...'.format(print_block_len))
        titles.append(row[0])
        abstracts.append(row[1])
    cursor.close()
    end_time = time.time()
    elapsed = end_time - start_time
    print("SQL results fetched and cast in {} s".format(elapsed))

    # create the 'corpus' := sentences list. each sentence is a words list.
    corpus = []
    for t,a in zip(titles, abstracts):

        P_ = t +  ' ' + a   # note, there is already a period at the end of the title
        # todo consider segmenting sentences - currently title and abstract are one long sample

        p_ = P_.lower()
        
        p = re.sub(r'[+-]?[\d]+',' _number_ ', p_)  # first sub numbers
        p = re.sub(r'[^\w\s]',' ',p)  # then remove punctuation
        p_clean = unicodedata.normalize('NFKD',p)
        
        words_clean = str.split(p_clean,' ')
        corpus.append(words_clean)
        
    # todo write corpus
    with open(save_path,'w') as f:
        print('saving new work to {}'.format(save_path))
        json.dump(corpus, f)



