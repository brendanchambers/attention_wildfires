#  encode the validation set

'''
Prototype a few options for encoding,
here using the vanilla BERT model.
'''

import spacy
is_using_gpu = spacy.prefer_gpu()
from spacy.util import minibatch
import random
import torch
from spacy_pytorch_transformers import PyTT_Language, PyTT_WordPiecer, PyTT_TokenVectorEncoder
import numpy as np
import cupy as cp
import sys
import time
import csv

###################
# verify gpu is active

print("GPU sanity checks: ")
print("device ID: {}".format(torch.cuda.current_device()))
print(torch.cuda.device(0))
print('N devices: {}'.format(torch.cuda.device_count()))
print(torch.cuda.get_device_name(0))
print("is available? {}".format(torch.cuda.is_available()))


###################
# Configure Spacy for vanilla BERT

start_time = time.time()

print("Spacy GPU? {}".format(is_using_gpu))
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

name = "scibert-scivocab-uncased"
path = "/project2/jevans/brendan/pretrained_transformers/scibert-scivocab-uncased"
nlp = PyTT_Language(pytt_name=name, meta={"lang": "en"})
nlp.add_pipe(nlp.create_pipe("sentencizer"))
nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, path))
nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, path))

end_time = time.time()
print('elapsed (s): {}'.format(end_time - start_time))

####################
# Encode abstracts from the validation set

start_time = time.time()

chunk_filepath = '/project2/jevans/brendan/pubmed_data_processing/validation_sets/'
chunk_filename = chunk_filepath + 'jneurophysiol_vs_neuroimage.csv'
embedding_subchunk_size = 10000  # do 10000 abstracts at a time

embedding_target_path = '/project2/jevans/brendan/pubmed_data_processing/validation_sets/jneurophysiol_vs_neuroimage_results/'
embedding_flavor = 'scibert__cls'

data = []
with open(chunk_filename) as f:

    csvreader = csv.reader(f, delimiter=' ')
    header = next(csvreader) # skip header
    print("header: {}".format(header))

    for idx, row in enumerate(csvreader):
        data.append(row)

        if idx % embedding_subchunk_size == 0:

            print('embedding subchunk number {}...'.format(idx / embedding_subchunk_size))

            # perform embedding:
            ###########################
            # remove rows with no text
            del_idxs = []
            for idx, row in enumerate(data):  # don't embed empty text
                if row[2] == '' and row[1] == '':
                    del_idxs.append(idx)
            for idx in sorted(del_idxs, reverse=True):
                del data[idx]

            abstracts = [t[1] + '. ' + t[2] for t in data]  # (pmid, title, abstract)

            end_time = time.time()
            print('elapsed (s): {}'.format(end_time - start_time))

            ######################
            # Process Batch of Abstracts
            print('processing batch of abstracts...')

            start_time = time.time()

            entries = []

            for idx, doc in enumerate(nlp.pipe(abstracts)):
                
                ############ control the representation here ##############
                last_hidden_state = doc._.pytt_last_hidden_state
                embedding_cupy = last_hidden_state[0,:]
                embedding = cp.asnumpy(embedding_cupy)
                ###########################################################
                              
                pmid = data[idx][0]
                entries.append((pmid, embedding.tolist(),))

            end_time = time.time()
            print("elapsed: {}".format(end_time - start_time))

            ########################

            print('writing embeddings to csv file...')
            start_time = time.time()
            
            with open(embedding_target_path + embedding_flavor + '.csv', 'w') as f:
                csv_out = csv.writer(f, delimiter=' ')
                for row_ in entries:
                    csv_out.writerow(row_)

            end_time = time.time()
            print("elapsed: {}".format(end_time - start_time))

            # reset data
            data = []

