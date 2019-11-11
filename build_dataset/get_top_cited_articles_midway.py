
'''
quick and rough script to get pmids for published articles by year, and the articles cited that year
'''

# play with some sql selections to make sure things are working

from pytorch_pretrained_bert import BertModel, BertTokenizer
import torch

import time
import mysql.connector as mysql

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import Row, StructType, StructField, IntegerType, StringType

import os
import time
import random
#import mysql.connector as mysql   # import gc




import csv

#import igraph

import time
import numpy as np

import matplotlib.pyplot as plt
#import cairocffi as cairo

import seaborn as sns
import umap
import statsmodels.api as sm  # for kdemultivariate


########
SUBMIT_ARGS = "--driver-class-path file:///home/brendanchambers/my_resources/mysql-connector-java-8.0.16/mysql-connector-java-8.0.16.jar --jars file:///home/brendanchambers/my_resources/mysql-connector-java-8.0.16/mysql-connector-java-8.0.16.jar pyspark-shell"
os.environ["PYSPARK_SUBMIT_ARGS"] = SUBMIT_ARGS

db_name = 'test_pubmed'  # db name collisons? https://stackoverflow.com/questions/14011968/user-cant-access-a-database
table_name = 'abstracts' # 'abstracts'

url = "jdbc:mysql://localhost:3306/{}?useUnicode=true&useJDBCCompliantTimezoneShift=true&useLegacyDatetimeCode=false&serverTimezone=America/Chicago".format(db_name)  # mysql runs on port 3306


data_dir = '/project2/jevans/brendan/open_citation/'  # this is 2019
open_cite_path = data_dir + 'open_citation_collection_2019-04.csv'
metadata_path = data_dir + 'icite_metadata_2019-04.csv'
#######
print('initializing spark')
# init spark
conf = SparkConf()
conf = (conf.setMaster('local[*]')
       .set('spark.driver.memory','28G')
       .set("spark.jars", "/home/brendanchambers/my_resources/mysql-connector-java-8.0.16/mysql-connector-java-8.0.16.jar"))
'''
.set('spark.executor.memory','1G')  # 20
.set('spark.driver.memory','1G')   # 40
.set('spark.driver.maxResultSize','500M')  #.set('spark.storage.memoryFraction',0))  # this setting is now a legacy option
.set('spark.python.worker.reuse', 'false')
.set('spark.python.worker.memory','512m')
.set('spark.executor.cores','1'))
'''
sc = SparkContext(conf=conf)
#sc.addJar('home/brendanchambers/my_resources/mysql-connector-java-8.0.16/mysql-connector-java-8.0.16.jar')  # temp
spark = SparkSession(sc)  # don't need this for vanilla RDDs

print(sc._conf.getAll())
######

# load in open citation dataset

edgelist = spark.read.csv(open_cite_path, header=True)
metadata = spark.read.csv(metadata_path, header=True)

# see schema
edgelist.printSchema()
metadata.printSchema()

######

# for many years


# for estimating running time -
# 10 s to get the pmids after caching (1959)
# 136 s to get the initial articles df (1960)  # todo use sql w indexing to speed this up?


start_year = 1959  # inclusive
end_year = 2020  # exclusive
articles_data = {}

for year in range(start_year, end_year):
    start_time = time.time()
    print(year)
    year_set = [year]  # todo is there function that doesn't require a list

    articles = metadata.filter(metadata.year.isin(year_set)).cache()

    articles_count = articles.count()
    print("articles count: {}".format(articles_count))

    year_pmids = articles.rdd.map(lambda row: row['pmid']).collect()
    articles.unpersist()

    # get cited articles
    year_pmids_broadcast = sc.broadcast(year_pmids)
    citation_count = edgelist.where(edgelist['referenced'].isin(year_pmids_broadcast.value)) \
        .select('referenced').groupBy('referenced').count().collect()

    citations_rdd = citation_count.rdd.cache()
    cited_pmids = citations_rdd.map(lambda row: row['referenced']).collect()
    cited_count = citations_rdd.map(lambda row: row['count']).collect()

    # todo write data

    articles_data[year] = {'N': articles_count,
                           'published_pmids': year_pmids,
                           'cited_pmids': cited_pmids,
                           'citation_count': cited_count}

    target_filename = 'json/articles_data_{}.json'.format(year)
    with open(target_filename, 'w') as f:
        json.dump(articles_data[year], f)

    end_time = time.time()
    print("elapsed: {}".format(end_time - start_time))




