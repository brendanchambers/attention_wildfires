Note Jan 2020

As it turns out,
this directory is for the most part all about the year 2018,
as I prototype analysis methods
for understanding changes across time.
So research_dynamics is a bit of a misnomer, this analysis is about a single year
todo reorganize at some point

_______________________________________________
current version Jan 30 2020





## data files



## library files

    pmids2vec.py

    pmids2corpus.py


##  notebook files

try_2018
    check the collection of publications in this year
    take 2 samples of size k
    cluster articles using hdbscan for each sample
    save the pmids and their respective clusters (export as json)
    
run_pmids2vec
    take the clustered pmids generated in try_2018
    train a word2vec model for each
    also export the corresponding corpuses generated based on each pmid cluster
                (for use in nulls and other analysis)
       depends on pmids2vec.py and pmids2corpus.py

explain_clusters
    load word2vec models trained on multiple clusters (research worlds)
    look at the semantic network for each world
    use high eig central words to characterize each world

try_CCA
    take 2 research worlds
    align their semantic networks using CCA
    take copy-and-pasted high eigenvector words (based on explain clusters)
    and compare their location across the aligned embeddings
    
check_null_CCA
    how much alignment is expected in the absense of meaningful interword semantics
      for 2 research worlds
    
compare_overlapping_worlds
    similar to explain_clusters.ipynb, but for only words appearing frequently across each research world
    


















_____________________________________________

older readme:


# documentation


try_2018
    takes -
    database
    pretrained umap reducer
    
    creates - 
    pmids [ sample ] [ cluster]
    summary_coords [ sample ] [ cluster]
    
run_pmids2vec
    train word2vec model for a group of pmids
    calls pmids2vec & pmids2corpus helper functions
    

compare overlapping worlds   (transform clustering to word2vec model)
    compare the word2vec models trained on the 2018 clusters - shared words only
    newer version of compare worlds
    
explain_clusters
    qualify character of each big cluster
    
    
scratch (eventually delete these):

try_network_viz

try_cluster_hotspots

sandbox

compare worlds   (transform clustering to word2vec model)
    compare the word2 vec models trained on the 2018 clusters
    
    
______________________

older:

yearstep_by_sampling

Sample papers and citations from each year.
    Fixed sample size?
    Or fraction of total year output
Link to their pre-computed representations.


