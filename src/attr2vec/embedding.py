from src.attr2vec.attr2vec import Attr2Vec
from gensim.models import KeyedVectors
from src.util.key_words import tfidf, graph_tf_idf
import os
import json

MODEL_PATH = 'model/vectors'


def attributed_embedding(
    G,
    d=8,
    walklen=30,
    epochs=20,
    return_weight=1,
    neighbor_weight=1,
    attribute_weight=1,
    seed=None,
    virtual_nodes=[],
    graph_name=None,
    get_weights=True
):
    # 文件名
    model_file_name = MODEL_PATH + "/" + "{}-d{}-wl{}-ep{}-rw{}-nw{}-aw{}-s{}".format(
        graph_name, d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed
    ).replace(".", "_").replace("/", "_")
    vector_file_name = model_file_name + ".bin"
    is_exist = False
    walks = None
    # if graph_name:
    # if os.path.exists(vector_file_name) and \
    # (not get_weights or os.path.exists(tfidf_file_name)):
    # is_exist = True
    if graph_name:
        if os.path.exists(vector_file_name):
            is_exist = True
    if False:
        print("Trained model {} exists. Skip training.".format(model_file_name))
    else:
        # 训练Attr2vec模型
        g2v = Attr2Vec(
            n_components=d,
            walklen=walklen,
            epochs=epochs,
            return_weight=return_weight,
            neighbor_weight=neighbor_weight,
            attribute_weight=attribute_weight,
            threads=2,
            virtual_nodes=virtual_nodes,
            seed=seed,
            keep_walks=True
        )
        g2v.fit(G)
        # 保存向量
        g2v.save_vectors(vector_file_name)
        walks = g2v.walks
        '''
        if True:
            # 计算tfidf权重
            pathes = []
            for walk in walks:
                walk_str = ' '.join(walk)
                pathes.append(walk_str)
            keyword_res = tfidf(pathes)
            # 保存权重数据
            with open(tfidf_file_name, 'w') as f:
                json_str = json.dumps(keyword_res)
                f.write(json_str)
        '''
    # 读取向量
    model = KeyedVectors.load_word2vec_format(vector_file_name)
    nodes = G.nodes
    vectors = {}
    for node in nodes:
        vectors[node] = model[str(node)]
    # 读取权重
    '''
    node_weights = {}
    if get_weights:
        with open(tfidf_file_name, 'r') as f:
            json_str = f.read()
            node_weights = json.loads(json_str)
    # 输出权重前十位
    node_weight_list = [(key, node_weights[key]) for key in node_weights.keys()]
    sorted_node_list = sorted(node_weight_list, key=lambda d: d[1], reverse=True)
    print(sorted_node_list[:10])
    '''
    # return vectors, node_weights
    return vectors, walks
