import pickle
import networkx as nx
import json
import numpy as np
from src.util.graph_reading import (
    read_citeseer_graph,
    read_cora_graph,
    read_facebook_graph,
    read_science_graph,
    read_influence_graph,
    read_security_graph,
    largest_connected_subgraph,
    read_cornell_graph,
)


def main():
    graphs = {
        "citeseer": read_citeseer_graph,
        # "cora": read_cora_graph,
        # "facebook": read_facebook_graph,
        # "influence": read_influence_graph,
        # "security": read_security_graph,
        # "cornell": read_cornell_graph,
    }
    for graph, reader in graphs.items():
        G, class_map = reader()
        LCG = largest_connected_subgraph(G)
        path = "data/lcg/"
        if graph in ["cornell"]:
            ph_path = path + graph + "_ph.json"
            save_for_ph(ph_path, LCG, class_map)
        pkl_path = path + graph + "_lcg.pkl"
        save_for_python(pkl_path, LCG, class_map)
        dr_path = path + graph + "_lcg_edge_list.txt"
        save_for_drgraph(dr_path, LCG, class_map)
        # graphtsne_path = path + graph + "_graphtsne_lcg.pkl"
        graphtsne_path = path + graph + "_graphtsne.pkl"
        if graph in ["citeseer", "cora", "cornell"]:
            save_cora_for_graphtsne(graphtsne_path, G, class_map)
        elif graph in ["security"]:
            save_for_graphtsne(graphtsne_path, LCG, class_map)


def save_for_drgraph(path, G, class_dict=None):
    edge_list = []
    for s, t in G.edges:
        edge_list.append((s, t))
    node_count = len(list(G.nodes))
    edge_count = len(edge_list)
    lines = []
    lines.append("{} {}".format(node_count, edge_count))
    for s, t in edge_list:
        lines.append("{} {} 1".format(s, t))
    with open(path, "w") as f:
        f.writelines(lines)


def save_cora_for_graphtsne(path, G, class_dict=None):
    nodes = list(G.nodes)
    feature_list = []
    label_list = []
    for node, label in class_dict.items():
        if label in label_list:
            continue
        label_list.append(label)
    for node in G.nodes(data=True):
        for feat, value in node[1].items():
            if feat not in feature_list:
                feature_list.append(feat)

    features = []
    labels = []
    for node in G.nodes(data=True):
        feats = []
        for feat in feature_list:
            if feat in node[1].keys():
                feats.append(1)
            else:
                feats.append(0)
        # attrs = list(node[1].values())
        features.append(feats)
        labels.append(label_list.index(class_dict[node[0]]))
    features = np.asarray(features, dtype='float32').astype('float32')
    labels = np.asarray(labels, dtype='int32').astype('int32')
    adj = nx.to_scipy_sparse_matrix(G, format='csr').astype('float32')
    with open(path, "wb") as f:
        pickle.dump([features, labels, adj], f)


def save_for_graphtsne(path, G, class_dict=None):
    nodes = list(G.nodes)
    feature_list = []
    for node in G.nodes(data=True):
        for feat, value in node[1].items():
            if feat not in feature_list:
                feature_list.append(feat)
    features = []
    labels = []
    for node in G.nodes(data=True):
        attrs = list(node[1].values())
        features.append(attrs)
        labels.append(class_dict[node[0]])
    features = np.asarray(features, dtype='float32').astype('float32')
    labels = np.asarray(labels, dtype='int32').astype('int32')
    adj = nx.to_scipy_sparse_matrix(G, format='csr').astype('float32')
    with open(path, "wb") as f:
        pickle.dump([features, labels, adj], f)


def save_for_python(path, G, class_dict=None):
    with open(path, "wb") as f:
        pickle.dump([G, class_dict], f)


def save_original():
    pass


def save_for_ph(path, G, class_dict):
    groups = []
    for node in G.nodes:
        clas = class_dict[node]
        if clas not in groups:
            groups.append(clas)
    pos = nx.fruchterman_reingold_layout(G, seed=6)
    ph_json = {}
    node_list = []
    edge_list = []
    for node in G.nodes:
        node_dict = {
            "id": node,
            "x": pos[node][0],
            "y": pos[node][1],
            "group": groups.index(class_dict[node])
        }
        node_list.append(node_dict)
    for s, t in G.edges:
        edge_dict = {
            "source": s,
            "target": t,
            "value": 1
        }
        edge_list.append(edge_dict)
    ph_json["nodes"] = node_list
    ph_json["links"] = edge_list
    with open(path, "w") as f:
        json.dump(ph_json, f)


if __name__ == "__main__":
    main()
