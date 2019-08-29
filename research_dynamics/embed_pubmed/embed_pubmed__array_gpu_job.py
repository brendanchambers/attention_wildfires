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

# get SLURM_ARRAY_TASK_ID
print('Input arguments: {}'.format(sys.argv))
slurm_array_id = int(sys.argv[1])
print('slurm array id: {}'.format(slurm_array_id))

###################
# check for gpu use

print("GPU sanity checks: ")
print("device ID: {}".format(torch.cuda.current_device()))
print(torch.cuda.device(0))
print('N devices: {}'.format(torch.cuda.device_count()))
print(torch.cuda.get_device_name(0))
print("is available? {}".format(torch.cuda.is_available()))

###################
# Configure Spacyls

start_time = time.time()

print("Spacy GPU? {}".format(is_using_gpu))
if is_using_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")

name = "scibert-scivocab-uncased"
path = "/project2/jevans/brendan/pretrained_transformers/scibert-scivocab-uncased"
# https://github.com/allenai/scibert
#  https://arxiv.org/abs/1903.10676

nlp = PyTT_Language(pytt_name=name, meta={"lang": "en"})
nlp.add_pipe(nlp.create_pipe("sentencizer"))
nlp.add_pipe(PyTT_WordPiecer.from_pretrained(nlp.vocab, path))
nlp.add_pipe(PyTT_TokenVectorEncoder.from_pretrained(nlp.vocab, path))

end_time = time.time()
print('elapsed (s): {}'.format(end_time - start_time))


#####################
# Sample Abstracts to Encode  (testing offset-limit pagination - how much penalty do we pay for table iteration?)
# todo do this in batches of 10000

start_time = time.time()

chunk_filepath = '/project2/jevans/brendan/pubmed_data_processing/parallelization_splits/'
chunk_filename = chunk_filepath + 'datachunk_{}.csv'.format(slurm_array_id-1)
embedding_subchunk_size = 10000  # do 10000 abstracts at a time in the 600000k files
data = []
with open(chunk_filename) as f:

    csvreader = csv.reader(f, delimiter=' ')
    next(csvreader) # skip header

    for idx, row in enumerate(csvreader):
        data.append(tuple(row))

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
                embedding_cupy = np.mean(last_hidden_state[1:,:], 0)
                embedding = cp.asnumpy(embedding_cupy)
                ###########################################################
                pmid = data[idx][0]
                entries.append((pmid, embedding.tolist(),))
                # entries.append( (embedding_blob,) )

            end_time = time.time()
            print("elapsed: {}".format(end_time - start_time))

            ########################
            # write batch of embeddings to file

            print('writing embeddings to csv file...')

            start_time = time.time()

            embedding_target_path = '/project2/jevans/brendan/pubmed_data_processing/scibert_embedding_chunks/'
            with open(embedding_target_path + 'chunk_{}.csv'.format(slurm_array_id), 'a') as f:
                csv_out = csv.writer(f, delimiter=' ')
                for row_ in entries:
                    csv_out.writerow(row_)

            end_time = time.time()
            print("elapsed: {}".format(end_time - start_time))


            # reset data
            data = []






