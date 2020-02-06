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




def pmids2vec(PMIDs, save_path):
    
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
        p = re.sub(r'[^\w\s]','',p_)  # todo check this
            #  todo entities? lemmatization? other preprocessing?
        # todo number string matching
        #   todo use significant digits to decide on the mask
        p_clean = re.sub(r'[ ][+-]?[\d]+',' <NUM>', p) # '^[-+]?[\d]$","\n",p) #  substitute numbers

        words_clean = str.split(p_clean,' ')
        corpus.append(words_clean)
        
    # train a word2vec model using these abstracts
    start_time = time.time()
    print('training word2vec model...')
    D = 20
    W = 5
    COUNT = 10
    print('params: {} dimensions, {} window size, {} min count'.format(D, W, COUNT))
    model = Word2Vec(corpus, size=D, window=W, min_count=COUNT, workers=16)
    end_time = time.time()
    print("elapsed: {}".format(end_time - start_time))

    # save the word2vec networks
    model.save(save_path)   # could just save the .kv file instead

def pmids2vec_titlesOnly(PMIDs, save_prefix):
    
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

    sql = '''SELECT A.title
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
    cursor.close()
    end_time = time.time()
    elapsed = end_time - start_time
    print("SQL results fetched and cast in {} s".format(elapsed))

    # create the 'corpus' := sentences list. each sentence is a words list.
    corpus = []
    for t in zip(titles):

        P_ = str(t)  # note, there is already a period at the end of the title
        # todo consider segmenting sentences - currently title and abstract are one long sample

        p_ = P_.lower()
        p = re.sub(r'[^\w\s]','',p_)  # todo check this
            #  todo entities? lemmatization? other preprocessing?
        # todo number string matching
        #   todo use significant digits to decide on the mask
        p_clean = re.sub(r'[ ][+-]?[\d]+',' <NUM>', p) # '^[-+]?[\d]$","\n",p) #  substitute numbers

        words_clean = str.split(p_clean,' ')
        corpus.append(words_clean)
        
    # write corpus
    savepath_corpus = save_prefix + '_titles__corpus.json'
    with open(savepath_corpus,'w') as f:
        print('saving corpus of titles to {}'.format(savepath_corpus))
        json.dump(corpus, f)
        
    # train a word2vec model using these titles
    start_time = time.time()
    print('training word2vec model...')
    D = 20
    W = 5
    COUNT = 10
    print('params: {} dimensions, {} window size, {} min count'.format(D, W, COUNT))
    model = Word2Vec(corpus, size=D, window=W, min_count=COUNT, workers=16)
    end_time = time.time()
    print("elapsed: {}".format(end_time - start_time))

    # save the word2vec networks
    savepath_w2v = save_prefix + '_titles.model'
    model.save(savepath_w2v)   # could just save the .kv file instead

if __name__ == '__main__':
    
     ####
      # dev set of pmids
    PMIDs_dir = '/home/brendan/FastData/pubmed2019/pubmed_data_processing/year_pmids/'
    PMIDs_name = 'pubmed_state_2008'  # these should have a .json postfix
    PMIDs_path = PMIDs_dir + PMIDs_name

    with open(PMIDs_path, 'r') as f:
        data = json.load(f)
        
    K = 100000  #  for testing - number of pmids to subselect
    PMIDs = data['publications'][:K] # testing, just grab the first K non-randomly
    print("N PMIDs: {}".format(len(PMIDs)))
    
    
    ####
    

    ####
    save_path = "test_word2vec.model"
    
    pmids2vec(PMIDs, save_path)