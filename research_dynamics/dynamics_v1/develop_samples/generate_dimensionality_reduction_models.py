# take some big samples of papers and generate reduced dimensional spaces suitable for use with
#   the entire dataset

import mysql.connector as mysql
import pickle
import json

import numpy as np
from sklearn.decomposition import PCA
import umap

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['svg.fonttype'] = 'none'

import time


#####

# path to save the generated pca models
model_output_path = '/project2/jevans/brendan/pubmed_data_processing/dimensionality_reduction_models/pca_models/'
umap_output_path = '/project2/jevans/brendan/pubmed_data_processing/dimensionality_reduction_models/umap2D/'

# mysql specs
db_name = 'test_pubmed'
client_config = {'unix_socket':'/home/brendanchambers/.sql.sock',
                'database': db_name,
                'use_pure': True}  # use the pure python connector (important for blob decoding)
db = mysql.connect(**client_config)

#####
# loop

N_samplesets = 10
for sampleset_id in range(N_samplesets):  # id for pregenerated sample pmids (in json files)

    ####
    # grab sample pmids

    path2dir = '/project2/jevans/brendan/pubmed_data_processing/sample_pmids/' # source for pre-generated ~100k sample sets
    filename = 'sample{}.json'.format(sampleset_id)
    path2pmids = path2dir + filename
    with open(path2pmids,'r') as f:
        data = json.load(f)

    print('overview:')
    print('sample fraction: {}'.format(data['sample_fraction']))
    print('resample? {} '.format(data['do_resample']))
    sample_pmids = [row[0] for row in data['sample_rows']]  # ignore year and journal info here
    print("n samples: {}".format(len(sample_pmids)))

    # format for mysql ingestion
    str_fmt = ', '.join([str(pmid) for pmid in sample_pmids])
    del sample_pmids
    del data #  don't need these anymore

    ####
    # match associated embeddings blobs on pmid
    sql = '''SELECT E.pmid, E.embedding
                FROM scibert_mean_embedding as E
                WHERE E.pmid IN ({})'''.format(str_fmt)
    start_time = time.time()
    cursor = db.cursor()
    cursor.execute(sql)
    output = cursor.fetchall()
    cursor.close()

    end_time = time.time()
    elapsed = end_time - start_time
    print("sql join executed in {} s".format(elapsed))
    print()

    ####
    # decode blobs to numpy float64 array
    sample_embeddings = np.array([np.frombuffer(row[1],dtype='float64') for row in output])
    print(np.shape(sample_embeddings)) # sanity check

    ####
    print('running principle component analysis...')

    start_time = time.time()

    D = 728
    D_pca = 728

    pca = PCA(n_components=D_pca)

    end_time = time.time()
    print("elapsed: {}".format(end_time - start_time))

    ####
    # save the pca model we just fit

    # Save to file in the current working directory
    pkl_filename = model_output_path + "pca_model_{}.pkl".format(sampleset_id)
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pca, file)
        
    

    ####
    # visualization

    # var explained
    plt.figure(figsize=(1,1))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title(model_output_path + 'PCA component var explained (sample {})'.format(sampleset_id))  
    plt.xlabel('component #')
    plt.ylabel('variance explained')

    plt.save('pca var explained {}.png'.format(sampleset_id))
    plt.save('pca var explained {}.svg'.format(sampleset_id))
    plt.show()

    # first 2 PCs
    pc = pca.fit_transform(sample_embeddings)

    plt.figure()
    sns.kdeplot(pc[:,0],pc[:,1])
    plt.title(model_output_path + 'first 2 PCs (sample {})'.format(sampleset_id))  

    plt.save('pca 2D {}.png'.format(sampleset_id))
    plt.save('pca 2D {}.svg'.format(sampleset_id))
    plt.show()

    # umap
    D_umap = 2
    reducer = umap.UMAP(n_components=D_umap)
    um = reducer.fit_transform(sample_embeddings)    # concatenated
    
    pkl_filename_umap = umap_output_path + "umap2D_{}.pkl".format(sampleset_id)
    with open(pkl_filename_umap, 'wb') as file:
        pickle.dump(reducer, file)  # save fitted model

    plt.figure()
    sns.kdeplot(um[:,0],um[:,1])
    plt.title(model_output_path + 'umap 2D (sample {})'.format(sampleset_id))  

    plt.save('umap 2D {}.png'.format(sampleset_id))
    plt.save('umap 2D {}.svg'.format(sampleset_id))
    plt.show()