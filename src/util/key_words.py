from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os
import json

MODEL_PATH = 'model/weights'


def tfidf(corpus):
    print("Start TF-IDF training...")
    tfidf = TfidfVectorizer(lowercase=False, token_pattern=r"\S+")
    tfidf_matrix = tfidf.fit_transform(corpus)
    feature_names = tfidf.get_feature_names()
    df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    result = {}
    for index, row in df.iteritems():
        name = str(row.name)
        value = 0
        row_list = list(row)
        for v in row_list:
            if v != 0:
                value = v
                break
        result[name] = value
    return result


def graph_tf_idf(
    G,
    node_group,
    walks,
    d=8,
    walklen=30,
    epochs=20,
    return_weight=1,
    neighbor_weight=1,
    attribute_weight=1,
    seed=None,
    graph_name=None
):
    # 文件名
    model_file_name = MODEL_PATH + "/" + "{}-d{}-wl{}-ep{}-rw{}-nw{}-aw{}-s{}".format(
        graph_name, d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed
    ).replace(".", "_").replace("/", "_")
    tfidf_file_name = model_file_name + "_tfidf.json"
    if not os.path.exists(tfidf_file_name):
        corpus = ['' for i in range(1 + max(node_group.values()))]
        for walk in walks:
            first_node = walk[0]
            if first_node.startswith("_"):
                continue
            walk_str = ' '.join(walk)
            corpus[node_group[first_node]
                   ] = corpus[node_group[first_node]] + walk_str + ' '
        tfV = TfidfVectorizer(lowercase=False, token_pattern=r"\S+")
        tfidf_matrix = tfV.fit_transform(corpus)
        feature_names = tfV.get_feature_names()
        df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
        result = {}
        for index, row in df.iteritems():
            name = str(row.name)
            row_list = list(row)
            if name.startswith("_"):
                continue
            g = node_group[name]
            result[name] = row_list[g]
        with open(tfidf_file_name, 'w') as f:
            json_str = json.dumps(result)
            f.write(json_str)
    result = {}
    with open(tfidf_file_name, 'r') as f:
        json_str = f.read()
        result = json.loads(json_str)
    return result
