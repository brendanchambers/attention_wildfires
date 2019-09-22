#  step through years and visualize in PCA space

import mysql.connector as mysql
import pickle
import json

import numpy as np
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['svg.fonttype'] = 'none'

import time

#######################
## mysql client

db_name = 'test_pubmed'  # db name collisons? https://stackoverflow.com/questions/14011968/user-cant-access-a-database
client_config = {'unix_socket':'/home/brendanchambers/.sql.sock',
                'database': db_name,
                'use_pure': True}  # for python connector
output_path = '/project2/jevans/brendan/pubmed_data_processing/year_pmids/'

########################
## load pre-fit pca models (fit using samples (size=100k) from the full corpus)

pca_path = 'develop_samples/pca_model1.pkl'  # more located at /project2...
with open(pca_path, 'rb') as file:
    pca_model = pickle.load(file)
    
#########################
## control params

temp_altstart = 2002 # todo delete temp_altstart
start_year = 1958
end_year = 2018
D_truncate = 300
path2dir = '/project2/jevans/brendan/pubmed_data_processing/year_pmids/'

###########################
## load year publication pmids & join to embeddings

process_pubs = False  # temporary
if process_pubs:
    year_pubs = {}
    for year in range(temp_altstart, end_year+1): # start_year, end_year+1):  # todo: use start_year, delete temp_altstart

        print('{}...'.format(year))

        db = mysql.connect(**client_config)

        filename = 'pubmed_state_{}'.format(year)
        path2pmids = path2dir + filename
        with open(path2pmids,'r') as f:
            data = json.load(f)

        year_pub_pmids = data['publications']
        N_pubs = len(year_pub_pmids)
        print("N pubs: {}".format(N_pubs))
        del data # clean up

        str_fmt = ', '.join([str(pmid) for pmid in year_pub_pmids])

        sql = '''SELECT E.pmid, E.embedding
                FROM scibert_mean_embedding as E
                WHERE E.pmid IN ({})'''.format(str_fmt)

        start_time = time.time()
        cursor = db.cursor(buffered=False)
        cursor.execute(sql)
        end_time = time.time()
        elapsed = end_time - start_time
        print("SQL join executed in {} s".format(elapsed))

        start_time = time.time()
        pub_embeddings = []
        pub_pmids = []
        for i,row in enumerate(cursor):
            print_block_len = 100000
            if i % print_block_len == 0:
                print('fetched {} rows...'.format(print_block_len))
            pub_pmids.append(row[0])
            pub_embeddings.append(np.frombuffer(row[1],dtype='float64').tolist())
        cursor.close()
        end_time = time.time()
        elapsed = end_time - start_time
        print("SQL results fetched and cast in {} s".format(elapsed))

        start_time = time.time()
        #year_pubs[year] = pca_model.transform(pub_embeddings)[:,:D_truncate]
        end_time = time.time()
        elapsed = end_time - start_time
        print("pca transform finished in {} s".format(elapsed))

        start_time = time.time()
        path = output_path + 'publication_embeddings/' + str(year) + '.json'
        save_obj = {'pmids': pub_pmids,
                    'embeddings': pub_embeddings}
        with open(path,'w') as f:
            json.dump(save_obj, f)
        end_time = time.time()
        elapsed = end_time - start_time
        print('finished writing output file in {} s...'.format(elapsed))

        print()

######################################
## load year citation pmids, join to embeddings, write to file

year_cites = {}

for year in range(temp_altstart, end_year+1):  # todo change back to start_year
    
    print('{}...'.format(year))
    
    db = mysql.connect(**client_config)

    filename = 'pubmed_state_{}'.format(year)
    path2pmids = path2dir + filename
    with open(path2pmids,'r') as f:
        data = json.load(f)
    
    year_cite_pmids = data['citations']
    del data # clean up
    N_citations = len(year_cite_pmids)
    print("N citations: {}".format(N_citations))
    
    str_fmt = ', '.join([str(pmid) for pmid in year_cite_pmids])
    
    sql = '''SELECT E.pmid, E.embedding
            FROM scibert_mean_embedding as E
            WHERE E.pmid IN ({})'''.format(str_fmt)
    
    start_time = time.time()
    cursor = db.cursor(buffered=False)
    cursor.execute(sql)
    end_time = time.time()
    elapsed = end_time - start_time
    print("SQL join executed in {} s".format(elapsed))

    start_time = time.time()
    cite_embeddings = []
    cite_pmids = []
    for i,row in enumerate(cursor):
        print_block_len = 100000
        if i % print_block_len == 0:
            print('fetched {} rows...'.format(print_block_len))
        cite_pmids.append(row[0])
        cite_embeddings.append(np.frombuffer(row[1],dtype='float64').tolist())

    cursor.close()
    print('fetched')

    end_time = time.time()
    elapsed = end_time - start_time
    print("SQL results fetched and cast in {} s".format(elapsed))

    start_time = time.time()
    #year_cites[year] = pca_model.transform(cite_embeddings)[:,:D_truncate]
    end_time = time.time()
    elapsed = end_time - start_time
    print("pca transform finished in {} s".format(elapsed))
        
    start_time = time.time()
    path = output_path + 'citation_embeddings/' + str(year) + '.json'
    save_obj = {'pmids': cite_pmids,
                'embeddings': cite_embeddings}
    with open(path,'w') as f:
        json.dump(save_obj, f)
    end_time = time.time()
    elapsed = end_time - start_time
    print('finished writing output file in {} s'.format(elapsed))
    
    db.close()
    print()
    
    

    
