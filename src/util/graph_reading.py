import pickle
import networkx as nx
import json
import csv
import os
import time
import random

from networkx.utils.decorators import preserve_random_state


def make_figure_path(graph_name, note, err_msg=None):
    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    time_str = time.strftime("%H-%M-%S", time.localtime(time.time()))
    save_path = fig_path + "/" + graph_name.replace(".", "_")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += "/" + time_str
    if note:
        save_path += "-" + note
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if err_msg is not None:
        err_log_file = save_path + "/error_message.txt"
        with open(err_log_file, "w", encoding="utf-8") as f:
            f.write(err_msg)
    return save_path


def read_ph_result(graph_name):
    res_path = "data/ph_{}.json".format(graph_name)
    pos = {}
    with open(res_path, "r") as f:
        text = f.read()
        json_graph = json.loads(text)
        nodes = json_graph["nodes"]
    for node in nodes:
        node_name = node['id'].replace(" ", "")
        x = node['x']
        y = node['y']
        pos[node_name] = (x, y)
    return pos


def read_drgraph_result(graph_name):
    nodemap_path = "data/drgraph/{}_nodemap.txt".format(graph_name)
    pos_path = "data/drgraph/{}.txt".format(graph_name)
    node_map = {}
    pos = {}
    with open(nodemap_path, "r") as f:
        for line in f.readlines():
            node = line.split()[0]
            index = int(line.split()[1])
            node_map[index] = node

    with open(pos_path, "r") as f:
        f.readline()
        index = 0
        for line in f.readlines():
            x = float(line.split()[0])
            y = float(line.split()[1])
            node = node_map[index]
            pos[node] = (x, y)
            index += 1
    return pos


def read_graphtsne_result(graph_name):
    nodes_path = "data/graphtsne/{}.content".format(graph_name)
    pos_path = "data/graphtsne/{}.pkl".format(graph_name)
    if graph_name == "citeseer":
        pos = {}
        G, node_class = read_citeseer_graph()
        with open(pos_path, 'rb') as f:
            pos_list = pickle.load(f)
        for i, node in enumerate(list(G.nodes)):
            pos[node] = [pos_list[i][0], pos_list[i][1]]
        return pos
    else:
        node_map = {}
        pos = {}
        with open(nodes_path, 'r') as f:
            lines = f.readlines()
            index = 0
            for line in lines:
                strs = line.split("\t")
                node_map[index] = strs[0]
                pos[strs[0]] = (0, 0)
                index += 1
        with open(pos_path, 'rb') as f:
            pos_list = pickle.load(f)
            index = 0
            for p in pos_list:
                pos[node_map[index]] = [p[0], p[1]]
                index += 1
        return pos

def read_graphtpp_result(graph_name):
    view_file_path = "data/graphtpp/{}.view".format(graph_name)
    pos = {}
    lines = []
    with open(view_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        strs = line.split(",")
        node = strs[0]
        pos_x = float(strs[2])
        pos_y = float(strs[3])
        pos[node] = (pos_x, pos_y)
    return pos

def read_influence_graph():
    node_file = "data/influence/painters_attributes.arff"
    edge_file = "data/influence/painters_edges.csv"
    G = nx.Graph()
    attributes = []
    lines = []
    with open(node_file, "r") as f:
        lines = f.readlines()
    node_start_tag = False
    for line in lines:
        if node_start_tag:
            strs = line.split(",")
            node = strs[-1].strip()
            attr_dict = {}
            for i, attr in enumerate(attributes):
                attr_dict[attr] = strs[i]
            G.add_node(node, **attr_dict)
        if line.startswith("@attribute"):
            strs = line.split()
            attr = strs[1]
            format = strs[2]
            if attr == "class" or attr == "name":
                pass
            else:
                attributes.append(attr)
        if line.startswith("@data"):
            node_start_tag = True

    with open(edge_file, "r") as f:
        line = f.readline()
        while line:
            strs = line.split(";")
            source = strs[0]
            target = strs[1].strip()
            if not G.has_node(source) or not G.has_node(target):
                print("influence graph error")
            G.add_edge(source, target)
            line = f.readline()
    return G, None


def read_security_graph():
    node_file = "data/security/network_security_day406_attributes.arff"
    edge_file = "data/security/network_security_day406_edges.csv"
    G = nx.Graph()
    attributes = []
    lines = []
    node_class = {}
    classes = []
    with open(node_file, "r") as f:
        lines = f.readlines()
    node_start_tag = False
    for line in lines:
        if node_start_tag:
            strs = line.split(",")
            node = strs[-1].strip()
            clas = strs[-2]
            attr_dict = {}
            for i, attr in enumerate(attributes):
                attr_dict[attr] = strs[i]
            G.add_node(node, **attr_dict)
            node_class[node] = classes.index(clas)
        if line.startswith("@attribute"):
            strs = line.split()
            attr = strs[1]
            format = strs[2]
            if attr == "classification":
                format = " ".join(strs[2:])
                classes = format.strip("{").strip("}").split(",")
            if attr == "node":
                pass
            else:
                attributes.append(attr)
        if line.startswith("@data"):
            node_start_tag = True

    with open(edge_file, "r") as f:
        line = f.readline()
        while line:
            strs = line.split(";")
            source = strs[0]
            target = strs[1].strip()
            if not G.has_node(source) or not G.has_node(target):
                print("influence graph error")
            G.add_edge(source, target)
            line = f.readline()
    return G, node_class


def read_lcg_graph(graph_name):
    path = "data/lcg/{}_lcg.pkl".format(graph_name)
    with open(path, "rb") as f:
        pkl_data = pickle.load(f)
    return pkl_data[0], pkl_data[1]


def read_cora_lcg():
    return read_lcg_graph("cora")


def read_citeseer_lcg():
    return read_lcg_graph("citeseer")


def read_facebook_lcg():
    return read_lcg_graph("facebook")


def read_security_lcg():
    return read_lcg_graph("security")


def read_influence_lcg():
    return read_lcg_graph("influence")


def read_json_graph(file_name):
    G = nx.Graph()
    with open(file_name, "r") as f:
        text = f.read()
        json_graph = json.loads(text)
        nodes = json_graph["nodes"]
        links = json_graph["links"]
        for node in nodes:
            attr_dict = {}
            for key in node.keys():
                if key not in ["x", "y", "id"]:
                    attr_dict[key] = node[key]
            node_name = node["id"].replace(" ", "")
            node_attrs = attr_dict
            G.add_node(node_name, **node_attrs)
            # G.add_node(node_name)
        for link in links:
            G.add_edge(link["source"].replace(" ", ""),
                       link["target"].replace(" ", ""))
    return G


'''
def read_facebook_graph(facebook_file="0"):
    feature_name_file = "data/facebook/{}.featnames".format(facebook_file)
    node_feature_file = "data/facebook/{}.feat".format(facebook_file)
    edge_file = "data/facebook/{}.edges".format(facebook_file)

    feature_list = []
    with open(feature_name_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            feature_content = strs[-1].strip()
            feature_name = "_".join(strs[1].split(";")[:-1])
            feature = (feature_name, "f" + feature_content)
            feature_list.append(feature)
    # print(feature_list)
    G = nx.Graph()
    with open(node_feature_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            node_id = strs[0]
            G.add_node(node_id)
            for i, f in enumerate(strs[1:]):
                if int(f) == 0:
                    continue
                feature = feature_list[i]
                G.nodes[node_id][feature[0]] = feature[1]
    with open(edge_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            s = strs[0]
            t = strs[1].strip()
            G.add_edge(s, t)
    return G
'''


def init_attributed_graph(G, skip=False):
    if skip:
        return []
    nodes = list(G.nodes(data=True))
    virtual_node_list = []
    for node in nodes:
        attr_dict = node[1]
        for attr_key in attr_dict.keys():
            virtual_node_name = (
                "_" + attr_key + "_" +
                str(attr_dict[attr_key]).replace(" ", "")
            )
            if virtual_node_name not in virtual_node_list:
                virtual_node_list.append(virtual_node_name)
                G.add_node(virtual_node_name)
            G.add_edge(node[0], virtual_node_name)
    return virtual_node_list


def clean_attributed_graph(G, vectors=None):
    nodes = list(G.nodes(data=True))
    for node in nodes:
        if isinstance(node[0], int):
            continue
        if node[0].startswith("_") or node[0].startswith("attri-"):
            G.remove_node(node[0])
            try:
                vectors.pop(node[0])
            except KeyError:
                continue


def get_degrees(G):
    nodes = G.nodes
    degrees = []
    for node in nodes:
        degrees.append(G.degree[node])
    return degrees


def largest_connected_subgraph(G):
    largest_component = max(nx.connected_components(G), key=len)
    LCG = nx.Graph()
    for s, t in G.edges:
        if s not in largest_component or t not in largest_component:
            continue
        LCG.add_edge(s, t)
    for n in LCG.nodes:
        ori_n = G.nodes[n]
        for key in ori_n.keys():
            LCG.nodes[n][key] = ori_n[key]
    return LCG


def read_chemical_disease_graph(file_path):
    chemical_vocab_file = file_path + "CTD_chemicals.csv"
    chemical_disease_file = file_path + "CTD_chemicals_diseases.csv"
    G = nx.Graph()
    chemical_vocabs = {}
    with open(chemical_vocab_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i in range(29):
            line = next(reader)
        for line in reader:
            chemical_id = line[1].strip("MESH:")
            parents = line[4].split("|")
            chemical_vocabs[chemical_id] = {}
            chemical_vocabs[chemical_id]['parents'] = []
            for s in parents:
                chemical_vocabs[chemical_id]['parents'].append(
                    s.strip("MESH:"))
            tree_nodes = line[5].split("|")
            chemical_vocabs[chemical_id]['treenodes'] = []
            for s in tree_nodes:
                try:
                    subs = s.split("/")[1]
                except IndexError:
                    continue
                chemical_vocabs[chemical_id]['treenodes'].append(
                    subs.strip("MESH:"))
    with open(chemical_disease_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i in range(29):
            line = next(reader)
        for line in reader:
            if line[5] == "":
                continue
            chemical_id = line[1].strip("MESH:")
            evidence = line[5]
            disease_id = line[4].strip("MESH:")
            G.add_node(chemical_id)
            G.nodes[chemical_id][evidence] = disease_id
    node_list = list(G.nodes)
    for node in node_list:
        parents = chemical_vocabs[node]["parents"]
        treenodes = chemical_vocabs[node]["treenodes"]
        for p in parents:
            if p not in node_list:
                continue
            G.add_edge(node, p)
        for tn in treenodes:
            if tn not in node_list:
                continue
            G.add_edge(node, p)
    return G


def read_cites_contents_graph(graph_name, with_class=False):
    G = nx.Graph()
    node_class = {}
    cite_file = "data/{}.cites".format(graph_name)
    content_file = "data/{}.content".format(graph_name)
    feature_dict = {}
    with open(content_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split("\t")
            node = strs[0]
            G.add_node(node)
            paper_class = strs[-1].strip()
            node_class[node] = paper_class
            # if with_class:
            #     G.nodes[node]["class"] = paper_class
            features = strs[1:-1]
            for i, feat in enumerate(features):
                # continue
                # TODO
                if feat == "0":
                    continue
                feat_name = "feat{}".format(i)
                # feat_name = "feat_{}_{}".format(i, feat)
                if feat_name not in feature_dict.keys():
                    feature_dict[feat_name] = 1
                else:
                    feature_dict[feat_name] += 1
                G.nodes[node][feat_name] = 1
    with open(cite_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split("\t")
            s = strs[0]
            t = strs[1].strip()
            if s not in node_class.keys():
                continue
            if t not in node_class.keys():
                continue
            G.add_edge(s, t)
    self_edge = []
    for n, nbrs in G.adjacency():
        for nbr in nbrs.keys():
            if n == nbr:
                self_edge.append(n)
    for s in self_edge:
        G.remove_edge(s, s)
        # print("remove {} to {}".format(s, s))
    G.remove_nodes_from(list(nx.isolates(G)))
    sorted_items = sorted(feature_dict.items(),
                          key=lambda d: (d[1]), reverse=True)
    key_features = [feat for feat, num in sorted_items]
    for node in list(G.nodes(data=True)):
        node_name = node[0]
        attr_dict = node[1]
        for feat in list(attr_dict.keys()):
            if feat in key_features:
                continue
            else:
                G.nodes[node_name].pop(feat)
    if with_class:
        for node in G.nodes:
            G.nodes[node]["class"] = node_class[node]
    # G = largest_connected_subgraph(G)
    print("Read {}, {} nodes, {} edges.".format(
        graph_name,
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, node_class


def read_cora_graph(with_class=False):
    return read_cites_contents_graph("cora", with_class)


def read_citeseer_graph(with_class=False):
    return read_cites_contents_graph("citeseer", with_class)


def read_ppi_graph():
    graph_name = "PP-Pathways_ppi"
    graph_file = "data/{}.csv".format(graph_name)
    G = nx.Graph()
    with open(graph_file, "r") as f:
        for line in f:
            strs = line.split(",")
            G.add_edge(strs[0], strs[1].strip())
    self_edge = []
    for n, nbrs in G.adjacency():
        for nbr in nbrs.keys():
            if n == nbr:
                self_edge.append(n)
    for s in self_edge:
        G.remove_edge(s, s)
    G.remove_nodes_from(list(nx.isolates(G)))
    print("Read {}, {} nodes, {} edges.".format(
        graph_name,
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, None


def read_ppt_graph():
    graph_name = "PPT-Ohmnet_tissues-combined"
    graph_file = "data/{}.edgelist".format(graph_name)
    tissue = "urinary_bladder"
    G = nx.Graph()
    with open(graph_file, "r") as f:
        f.readline()
        for line in f:
            strs = line.split("\t")
            if strs[2].strip() != tissue:
                continue
            G.add_edge(strs[0], strs[1])
    self_edge = []
    for n, nbrs in G.adjacency():
        for nbr in nbrs.keys():
            if n == nbr:
                self_edge.append(n)
    for s in self_edge:
        G.remove_edge(s, s)
    # G.remove_nodes_from(list(nx.isolates(G)))
    print("Read {}, {} nodes, {} edges.".format(
        graph_name,
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, None


def read_miserables_graph():
    G = read_json_graph("data/miserables.json")
    print("Read {}, {} nodes, {} edges.".format(
        "Miserables",
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    class_map = {}
    for node in G.nodes(data=True):
        node_name = node[0]
        attrs = node[1]
        class_map[node_name] = attrs['group']
    return G, class_map


def read_science_graph():
    G = read_json_graph("data/map.science.json")
    print("Read {}, {} nodes, {} edges.".format(
        "Science",
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    class_map = {}
    for node in G.nodes(data=True):
        node_name = node[0]
        attrs = node[1]
        class_map[node_name] = attrs['group']
    return G, class_map


def read_facebook_graph():
    feature_name_file = "data/facebook/0.featnames"
    node_feature_file = "data/facebook/0.feat"
    edge_file = "data/facebook/0.edges"

    feature_list = []
    with open(feature_name_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            feature_content = strs[-1].strip()
            feature_name = "_".join(strs[1].split(";")[:-1])
            feature = (feature_name, "f" + feature_content)
            feature_list.append(feature)
    # print(feature_list)
    G = nx.Graph()
    with open(node_feature_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            node_id = strs[0]
            G.add_node(node_id)
            for i, f in enumerate(strs[1:]):
                if int(f) == 0:
                    continue
                feature = feature_list[i]
                G.nodes[node_id][feature[0]] = feature[1]
    with open(edge_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            s = strs[0]
            t = strs[1].strip()
            G.add_edge(s, t)
    # 保留最大连通子图
    # G = largest_connected_subsubgraph(G)
    print("Read {}, {} nodes, {} edges.".format(
        "Facebook",
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, None

def read_facebook_graph_numeric():
    feature_name_file = "data/facebook/0.featnames"
    node_feature_file = "data/facebook/0.feat"
    edge_file = "data/facebook/0.edges"

    feature_list = []
    with open(feature_name_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            feature_content = strs[-1].strip()
            feature_name = "_".join(strs[1].split(";")[:-1])
            feature = (feature_name, "f" + feature_content)
            feature_list.append(feature)
    # print(feature_list)
    G = nx.Graph()
    with open(node_feature_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            node_id = strs[0]
            G.add_node(node_id)
            for i, f in enumerate(strs[1:]):
                if int(f) == 0:
                    continue
                feature = feature_list[i]
                G.nodes[node_id][feature[0]+"_"+feature[1]] = 1
    with open(edge_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            s = strs[0]
            t = strs[1].strip()
            G.add_edge(s, t)
    # 保留最大连通子图
    # G = largest_connected_subsubgraph(G)
    print("Read {}, {} nodes, {} edges.".format(
        "Facebook",
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, None



def read_nature150_graph():
    graph_name = "Nature150"
    edge_file = "data/cociteEdges.csv"
    node_file = "data/cociteNodes.csv"
    G = nx.Graph()
    class_map = {}
    with open(node_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = next(reader)
        for line in reader:
            paper_id = line[5]
            hiercat = int(line[4])
            class_map[paper_id] = hiercat
            G.add_node(paper_id)
            G.nodes[paper_id]['hiercat'] = hiercat
    with open(edge_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        line = next(reader)
        for line in reader:
            source = line[0]
            target = line[1]
            G.add_edge(source, target)
    print("Read {}, {} nodes, {} edges.".format(
        graph_name,
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, class_map


def generate_random_graph():
    d = 1
    class_map = {}
    G = nx.sedgewick_maze_graph()
    strG = nx.Graph()
    for s, t in G.edges:
        strG.add_edge(str(s), str(t))
    G = strG
    for node in G.nodes(data=True):
        r = random.randint(0, d)
        node[1]['group'] = r
        class_map[node[0]] = r
    return G, class_map


def read_terrorist_attacks():
    path = "data/linqs.soe.ucsc.edu/terrorist-attacks/"
    node_file = path + "terrorist_attack.nodes"
    label_file = path + "terrorist_attack.labels"
    edge_file = path + "terrorist_attack_loc.edges"
    labels = []
    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            label = line.split("#")[1]
            labels.append(label)
    G = nx.Graph()
    node_label = {}
    feat_dict = {}
    with open(node_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split("\t")
            node = strs[0].split("#")[1]
            G.add_node(node)
            paper_class = strs[-1].strip().split("#")[1]
            node_label[node] = paper_class
            # if with_class:
            #     G.nodes[node]["class"] = paper_class
            features = strs[1:-1]
            for i, feat in enumerate(features):
                # continue
                # TODO
                if feat == "0":
                    continue
                feat_name = "feat{}".format(i)
                # feat_name = "feat_{}_{}".format(i, feat)
                if feat_name not in feat_dict.keys():
                    feat_dict[feat_name] = 1
                else:
                    feat_dict[feat_name] += 1
                G.nodes[node][feat_name] = 1
    with open(edge_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            s = strs[0].split("#")[1]
            t = strs[1].strip().split("#")[1]
            if s not in node_label.keys():
                print("node not found")
                continue
            if t not in node_label.keys():
                print("node not found")
                continue
            G.add_edge(s, t)
    # G.remove_nodes_from(list(nx.isolates(G)))
    # sorted_items = sorted(feature_dict.items(), key=lambda d: (d[1]), reverse=True)
    # key_features = [feat for feat, num in sorted_items]
    # for node in list(G.nodes(data=True)):
    #     node_name = node[0]
    #     attr_dict = node[1]
    #     for feat in list(attr_dict.keys()):
    #         if feat in key_features:
    #             continue
    #         else:
    #             G.nodes[node_name].pop(feat)
    # if with_class:
    #     for node in G.nodes:
    #         G.nodes[node]["class"] = node_class[node]
    # G = largest_connected_subgraph(G)
    print("Read {}, {} nodes, {} edges.".format(
        "terrorist attacks",
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, node_label


def read_webkb_graph(name):
    path = "data/webkb/"
    node_file = path + "{}.content".format(name)
    edge_file = path + "{}.cites".format(name)
    G = nx.Graph()
    node_label = {}
    feat_dict = {}
    with open(node_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split("\t")
            node = strs[0]
            G.add_node(node)
            paper_class = strs[-1].strip()
            node_label[node] = paper_class
            # if with_class:
            #     G.nodes[node]["class"] = paper_class
            features = strs[1:-1]
            for i, feat in enumerate(features):
                # continue
                # TODO
                if feat == "0":
                    continue
                feat_name = "feat{}".format(i)
                # feat_name = "feat_{}_{}".format(i, feat)
                if feat_name not in feat_dict.keys():
                    feat_dict[feat_name] = 1
                else:
                    feat_dict[feat_name] += 1
                G.nodes[node][feat_name] = 1
    with open(edge_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(" ")
            s = strs[0]
            t = strs[1].strip()
            if s not in node_label.keys():
                print("node not found")
                continue
            if t not in node_label.keys():
                print("node not found")
                continue
            G.add_edge(s, t)
    # G.remove_nodes_from(list(nx.isolates(G)))
    # sorted_items = sorted(feature_dict.items(), key=lambda d: (d[1]), reverse=True)
    # key_features = [feat for feat, num in sorted_items]
    # for node in list(G.nodes(data=True)):
    #     node_name = node[0]
    #     attr_dict = node[1]
    #     for feat in list(attr_dict.keys()):
    #         if feat in key_features:
    #             continue
    #         else:
    #             G.nodes[node_name].pop(feat)
    # if with_class:
    #     for node in G.nodes:
    #         G.nodes[node]["class"] = node_class[node]
    # G = largest_connected_subgraph(G)
    print("Read {}, {} nodes, {} edges.".format(
        name,
        len(list(G.nodes)),
        len(list(G.edges))
    ))
    return G, node_label


def read_cornell_graph():
    return read_webkb_graph("cornell")
