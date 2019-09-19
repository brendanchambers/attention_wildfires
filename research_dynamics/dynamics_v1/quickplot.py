#  step through years and visualize in pca & umap space

import mysql.connector as mysql
import pickle
import json

import numpy as np
from sklearn.decomposition import PCA
import umap

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['svg.fonttype'] = 'none'

import 

######

top_level_path = '/project2/jevans/brendan/pubmed_data_processing/year_pmids/'
pubs_dir = top_level_path + 'publication_embeddings/'
cites_dir = top_level_path + 'citation_embeddings/'

######

pca_path ="/project2/jevans/brendan/pubmed_data_processing/dimensionality_reduction_models/pca_models/pca_model0.pkl"
with open(pca_path, 'rb') as file:
    pca_model = pickle.load(file)

umap4pca_path = "/project2/jevans/brendan/pubmed_data_processing/dimensionality_reduction_models/umap2D/umap_model0_forPCA.pkl"
# this was fit with 300D pca vectors which can be regenerated using ../pca_models/pca_model0.pkl
with open(umap4pca_path, 'rb') as file:
    umap_model = pickle.load(file)
    
######
# helper function for the two dimensionality reduction density plots

# simplified version - for one year
def plot_pubs_and_cites(): 
    (f, ax) = plt.subplots(1,
                       2,
                       sharex='all', sharey='all',
                       figsize=(10,4))

    sns.kdeplot(pub_data['embeddings'][:,0], # these are pca'd
                pub_data['embeddings'][:,1],
                ax=ax[0],
                shade=True,
                cmap='Blues')
    ax[0].set_title('published: year {}'.format(year))


    sns.kdeplot(cite_data['embeddings'][:,0],
                cite_data['embeddings'][:,1],
                ax=ax[1],
                shade=True,
                cmap='Reds')
    ax[1].set_title('cited: {}'.format(year))

    plt.savefig('yearsteps/yearstep pca {} - {}.png'.format(start_year, end_year))
    plt.savefig('yearsteps/yearstep pca {} - {}.svg'.format(start_year, end_year))
    plt.show()
    
# umap version
def umap_pubs_and_cites():
    
    um_pubs = umap_model.transform(pub_data['embeddings'])
    um_cites = umap_model.transform(cite_data['embeddings'])
    
    (f, ax) = plt.subplots(1,
                       2,
                       sharex='all', sharey='all',
                       figsize=(10,4))

    sns.kdeplot(um_pubs[:,0], # these are pca'd
                um_pubs[:,1],
                ax=ax[0],
                shade=True,
                cmap='Blues')
    ax[0].set_title('published: year {}'.format(year))

    sns.kdeplot(um_cites[:,0],
                um_cites[:,1],
                ax=ax[1],
                shade=True,
                cmap='Reds')
    ax[1].set_title('cited: {}'.format(year))

    plt.savefig('yearsteps/yearstep umap4pca0 {} - {}.png'.format(start_year, end_year))
    plt.savefig('yearsteps/yearstep umap4pca0 {} - {}.svg'.format(start_year, end_year))
    plt.show()

######
# plot year state umap style

print(umap_model) 

# pre-fit manifold structure, from sample0 (representative corpus-wide set of pmids)
plt.figure()
sns.kdeplot(umap_model.embedding_[:,0], umap_model.embedding_[:,1])
plt.show()

start_year = 1958
end_year = 1999

for year in range(start_year, end_year+1):
    print(year)
    
    pubs_path = pubs_dir + str(year) + '.json'
    cites_path = cites_dir + str(year) + '.json'
    
    with open(pubs_path, 'r') as f:
        pub_data = json.load(f)   
    pub_data['embeddings'] = np.array(pub_data['embeddings'])
    #print(np.shape(pub_data['embeddings']))
    
    
    with open(cites_path, 'r') as f:
        cite_data = json.load(f)
    cite_data['embeddings'] = np.array(cite_data['embeddings'])
    
    umap_pubs_and_cites()

######
# plot yearstep pca style

start_year = 1958
end_year = 1999

for year in range(start_year, end_year+1):
    print(year)
    
    pubs_path = pubs_dir + str(year) + '.json'
    cites_path = cites_dir + str(year) + '.json'
    
    with open(pubs_path, 'r') as f:
        pub_data = json.load(f)   
    pub_data['embeddings'] = np.array(pub_data['embeddings'])
    #print(np.shape(pub_data['embeddings']))
    
    with open(cites_path, 'r') as f:
        cite_data = json.load(f)
    cite_data['embeddings'] = np.array(cite_data['embeddings'])
    
    plot_pubs_and_cites()
    


