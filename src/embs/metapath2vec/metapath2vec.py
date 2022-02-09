import json
import numpy as np
import os
from skipgram import build_model, train, traning_op
from genmetapaths import MetaPathGenerator
from dataset import Dataset
import sys
sys.path.append("./src/embs/metapath2vec/")


def mp2vec(walk_txt,
           node_type_txt,
           epochs=2,
           lr=0.01,
           d=100,
           window=1,
           negative_samples=5,
           care_type=1):
    index_file = "./model/mp2vec/index2nodeid.json"
    emb_file = "./model/mp2vec/node_embeddings.npz"
    if not os.path.exists(index_file) or not os.path.exists(emb_file):
        dataset = Dataset(random_walk_txt=walk_txt,
                          node_type_mapping_txt=node_type_txt, window_size=window)
        center_node_placeholder, context_node_placeholder, negative_samples_placeholder, loss = build_model(
            BATCH_SIZE=1, VOCAB_SIZE=len(dataset.nodeid2index), EMBED_SIZE=d, NUM_SAMPLED=negative_samples)
        optimizer = traning_op(loss, LEARNING_RATE=lr)
        print("training starts.")
        train(center_node_placeholder, context_node_placeholder, negative_samples_placeholder,
              loss, dataset, optimizer, NUM_EPOCHS=epochs, BATCH_SIZE=1, NUM_SAMPLED=negative_samples,
              care_type=care_type, LOG_DIRECTORY="./model/mp2vec", LOG_INTERVAL=-1,
              MAX_KEEP_MODEL=10)
    else:
        print("Embedding file exists. Skip training.")
    index2nodeid = json.load(open(index_file))
    index2nodeid = {int(k): v for k, v in index2nodeid.items()}
    nodeid2index = {v: int(k) for k, v in index2nodeid.items()}
    node_embeddings = np.load(emb_file)['arr_0']
    # node embeddings of "yi"
    embeidng_dict = {}
    for node in nodeid2index.keys():
        embeidng_dict[node] = node_embeddings[nodeid2index[node]]
    return embeidng_dict
