import networkx as nx
from .graph_reading import (
    read_cornell_graph,
    read_miserables_graph,
    read_cora_graph,
    read_citeseer_graph,
    read_science_graph,
    read_facebook_graph,
    read_facebook_graph_numeric
)
import os
import numpy as np
import pickle


readers = {
    # "miserables": read_miserables_graph,
    # "science": read_science_graph,
    # "facebook": read_facebook_graph,
    # "cora": read_cora_graph,
    # "citeseer": read_citeseer_graph,
    "cornell": read_cornell_graph,
}


def write_weka_graph(path, filename, weka_graph, node_class=None):
    if not os.path.exists(path):
        os.mkdir(path)
    node_dict = weka_graph["nodes"]
    attr_dict = weka_graph["attrs"]
    edge_list = weka_graph["edges"]
    classes = []
    if node_class != None:
        for node, clas in node_class.items():
            if clas not in classes:
                classes.append(clas)
    # arff file, containing node and attribute infomation
    node_file_path = path + "/" + filename + "_nodes.arff"
    with open(node_file_path, "w") as f:
        lines = []
        lines.append("@relation {}\n".format(filename.replace(" ", "_")))
        lines.append("@attribute NodeId string\n")
        if node_class:
            lines.append("@attribute class {" + ','.join(classes) + "}\n")
        else:
            lines.append("@attribute class {A}\n")
        for attr_key, attr_type in attr_dict.items():
            if attr_key == "group":
                enum_body = "{" + ",".join([str(i) for i in range(15)]) + "}"
                lines.append("@attribute {} {}\n".format(attr_key, enum_body))
            else:
                lines.append("@attribute {} {}\n".format(attr_key, attr_type))
        # lines.append("@attribute class {p}\n")
        lines.append("@data\n")
        for node, attrs in node_dict.items():
            line = ""
            line += "{},".format(node)
            if node_class != None:
                line += node_class[node] + ","
            else:
                line += "A,"
            for attr_key in attr_dict.keys():
                if attr_key in attrs.keys():
                    line += str(attrs[attr_key]) + ","
                else:
                    line += "0,"
            line = line[:-1]
            line += "\n"
            lines.append(line)
        f.writelines(lines)
    # csv file, containing node and attribute infomation
    node_file_path = path + "/" + filename + "_nodes.csv"
    with open(node_file_path, "w") as f:
        lines = []
        line = ""
        for attr_key, attr_type in attr_dict.items():
            line += attr_key + ","
        line += "nodeid\n"
        lines.append(line)
        for node, attrs in node_dict.items():
            line = ""
            for attr_key in attr_dict.keys():
                if attr_key in attrs.keys():
                    line += str(attrs[attr_key]) + ","
                else:
                    line += "0,"
            line += "{}\n".format(node)
            lines.append(line)
        f.writelines(lines)
    # edge file
    edge_file_path = path + "/" + filename + "_edges.csv"
    with open(edge_file_path, "w") as f:
        lines = []
        for s, t in edge_list:
            lines.append("{},{}\n".format(s, t))
        f.writelines(lines)


def nx_to_weka(G: nx.Graph):
    nodes = list(G.nodes(data=True))
    node_dict = {}
    attr_dict = {}
    edge_list = []
    for node in nodes:
        node_dict[node[0]] = {}
        attrs = node[1]
        for attr_key in attrs.keys():
            node_dict[node[0]][attr_key] = attrs[attr_key]
            if attr_key not in attr_dict.keys():
                attr_dict[attr_key] = "real"
    for s, t in G.edges:
        edge_list.append((s, t))
    
    return dict(nodes=node_dict, edges=edge_list, attrs=attr_dict)


def nx_to_edge_list(G: nx.Graph):
    nodes = list(G.nodes(data=True))
    node_dict = {}
    node_count = 0
    edge_list = []
    for node in nodes:
        if node[0] not in node_dict.keys():
            node_dict[node[0]] = node_count
            node_count += 1
    for s, t in G.edges:
        edge_list.append((node_dict[s], node_dict[t]))
    return node_dict, edge_list


def citeseer_to_graphtsne():
    id2index = {}
    label2index = {
        'Agents': 0,
        'IR': 1,
        'DB': 2,
        'AI': 3,
        'HCI': 4,
        'ML': 5,
    }
    features = []
    labels = []

    with open('./data/citeseer.content', 'r') as f:
        i = 0
        for line in f.readlines():
            items = line.strip().split('\t')

            id = items[0]

            # 1-hot encode labels
            label = label2index[items[-1]]
            # label = np.zeros(len(label2index))
            # label[label2index[items[-1]]] = 1
            labels.append(label)

            # parse features
            features.append([int(x) for x in items[1:-1]])

            id2index[id] = i
            i += 1

    features = np.asarray(features, dtype='float32').astype('float32')
    labels = np.asarray(labels, dtype='int32').astype('int32')

    graph = nx.Graph()
    bad_edge_count = 0
    with open('./data/citeseer.cites', 'r') as f:
        for line in f.readlines():
            items = line.strip().split('\t')
            try:
                tail = id2index[items[0]]
                head = id2index[items[1]]
            except KeyError:
                bad_edge_count += 1
                continue

            graph.add_edge(head, tail)
    print("bad edge " + str(bad_edge_count))
    adj = nx.to_scipy_sparse_matrix(graph, format='csr').astype('float32')

    with open("./data/citeseer_full.pkl", 'wb') as fo:
        pickle.dump([features, labels, adj], fo)


def write_edge_list(path, filename, node_dict, edge_list):
    if not os.path.exists(path):
        os.mkdir(path)
    nodemap_file_path = path + "/" + filename + "_edge_list_nodemap.txt"
    with open(nodemap_file_path, "w") as f:
        lines = []
        for node, id in node_dict.items():
            lines.append("{} {}\n".format(node, id))
        f.writelines(lines)
    edge_file_path = path + "/" + filename + "_edge_list.txt"
    with open(edge_file_path, "w") as f:
        lines = []
        lines.append("{} {}\n".format(
            len(list(node_dict.keys())), len(edge_list)))
        for s, t in edge_list:
            lines.append("{} {} 1\n".format(s, t))
        f.writelines(lines)


def all_to_weka():
    path = "data/weka"

    readers = {
        "cora": read_cora_graph,
        "citeseer": read_citeseer_graph,
        "miserables": read_miserables_graph,
        "science": read_science_graph,
        "facebook": read_facebook_graph_numeric,
        "cornell": read_cornell_graph
    }

    for graph_name, reader in readers.items():
        if graph_name in ["facebook", "cora", "citeseer", "cornell"]:
            G, node_class = reader()
        else:
            continue
            G = reader()[0]
        weka_graph = nx_to_weka(G)
        write_weka_graph(path, graph_name, weka_graph, node_class)


def all_to_edge_list():
    path = "data/edgelist"
    for graph_name, reader in readers.items():
        G = reader()[0]
        node_dict, edge_list = nx_to_edge_list(G)
        write_edge_list(path, graph_name, node_dict, edge_list)

