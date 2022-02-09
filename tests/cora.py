import networkx as nx
import matplotlib.pyplot as plt
from src.util.graph_reading import (
    init_attributed_graph,
    clean_attributed_graph
)
from src.util.normalize import normalize_list
from src.util.cluster import kmeans
from src.layout.embedding_fr import embedding_fr
from src.layout.reduction import tsne
from src.attr2vec.embedding import attributed_embedding
import time
import os

COLOR_MAP = {
    0: "tab:blue",
    1: "tab:brown",
    2: "tab:orange",
    3: "tab:pink",
    4: "tab:green",
    5: "tab:red",
    6: "tab:olive",
    7: "tab:purple",
    8: "tab:cyan",
    9: "limegreen",
    10: "teal",
    11: "tan",
    12: "salmon",
    13: "dodgerblue",
    14: "maroon",
}


def read_cora_graph(file_name):
    G = nx.Graph()
    node_class = {}
    cite_file = "data/{}.cites".format(file_name)
    content_file = "data/{}.content".format(file_name)
    feature_dict = {}
    with open(content_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split("\t")
            node = strs[0]
            G.add_node(node)
            paper_class = strs[-1].strip()
            node_class[node] = paper_class
            # G.nodes[node]["class"] = paper_class
            features = strs[1:-1]
            for i, feat in enumerate(features):
                # continue
                if feat == "0":
                    continue
                feat_name = "feat{}".format(i)
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
    print(len(list(G.nodes)))
    G.remove_nodes_from(list(nx.isolates(G)))
    print(len(list(G.nodes)))
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
                # continue
                G.nodes[node_name].pop(feat)
    return G, node_class


def draw_cora_embedding_fr(
        d=8,
        walklen=30,
        return_weight=1,
        neighbor_weight=1,
        attribute_weight=1,
        epochs=30,
        k=7,
        seed=None,
        color=None,
        size_min=8, size_max=8,
        width=0.5, edge_alpha=0.7,
        te=0.6, wa=1, we=1, note=""):
    graph_file = "cora"

    G, node_class = read_cora_graph(graph_file)
    class_map = {}
    count = 0
    for c in node_class.values():
        if c in class_map.keys():
            continue
        class_map[c] = count
        count += 1
    print("Embedding starts.")
    start = time.time()
    virtual_nodes = init_attributed_graph(G)
    vectors, weights = attributed_embedding(
        G,
        d=d,
        walklen=walklen,
        return_weight=1/return_weight,
        neighbor_weight=1/neighbor_weight,
        attribute_weight=1/attribute_weight,
        epochs=epochs,
        seed=seed,
        virtual_nodes=virtual_nodes,
        graph_name=graph_file,
        get_weights=False)
    clean_attributed_graph(G, vectors)

    date_path = time.strftime("%Y%m%d", time.localtime(time.time()))
    fig_path = "fig/" + date_path
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    time_str = time.strftime("%H-%M-%S", time.localtime(time.time()))
    save_path = fig_path + "/" + graph_file.replace(".", "_")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path += "/" + time_str + "-" + note
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    km_start = time.time()
    cluster_color = kmeans(vectors, K=None)
    km_end = time.time()
    print("K-Means time: {}s".format(round(km_end - km_start, 3)))
    cluster_color_list = []
    color_list = []
    for node in G.nodes:
        c = node_class[node]
        c_num = class_map[c]
        color_list.append(COLOR_MAP[c_num])
        cluster_color_list.append(COLOR_MAP[cluster_color[node]])
    tsne_pos = tsne(G, vectors)
    end = time.time()
    em_t = round(end - start, 3)
    plt.figure(figsize=(10, 10))
    plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}".format(
        "EM", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed
    )
    plt.title(fig_title)
    nx.draw_networkx_nodes(G, tsne_pos, node_size=15,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, tsne_pos, labels=labels, font_size=10)
    plt.savefig(
        save_path
        + "/{}-EM-{}s.png".format(graph_file.replace("/",
                                  "_").replace(".", "_"), em_t)
    )
    plt.close()

    start = time.time()
    fr_pos = nx.fruchterman_reingold_layout(G, seed=17)
    end = time.time()
    fr_t = round(end - start, 3)
    plt.figure(figsize=(10, 10))
    plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}".format(
        "FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed
    )
    plt.title(fig_title)
    nx.draw_networkx_nodes(G, fr_pos, node_size=15,
                           node_color=color_list, edgecolors='white', linewidths=0.7)
    # nx.draw_networkx_labels(G, fr_pos, labels=labels, font_size=10)
    nx.draw_networkx_edges(G, fr_pos, width=0.5, alpha=0.3)
    plt.savefig(
        save_path
        + "/{}-FR-{}s.png".format(graph_file.replace("/",
                                  "_").replace(".", "_"), fr_t)
    )
    plt.close()

    for i in [0.5]:
        for j in [0.8]:
            wa = i
            te = j

            start = time.time()
            em_fr_pos = embedding_fr(
                G, vectors=vectors, te=te, wa=wa, we=we, cluster=cluster_color)
            end = time.time()
            em_fr_t = round(end - start, 3)
            plt.figure(figsize=(10, 10))
            plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
            fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}-wa{}-we{}-te{}".format(
                "EM+FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed,
                wa, we, te
            )
            plt.title(fig_title)
            nx.draw_networkx_nodes(
                G, em_fr_pos, node_size=15, node_color=cluster_color_list, edgecolors='white', linewidths=0.7)
            # nx.draw_networkx_labels(G, em_fr_pos, labels=labels, font_size=10)
            nx.draw_networkx_edges(G, em_fr_pos, width=0.2, alpha=0.2)
            plt.savefig(
                save_path
                + "/{}-EM-FR-KMeans-{}s.png".format(
                    graph_file.replace("/", "_").replace(".", "_"), em_fr_t)
            )
            plt.close()
            plt.figure(figsize=(10, 10))
            plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
            fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}-wa{}-we{}-te{}".format(
                "EM+FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed,
                wa, we, te
            )
            plt.title(fig_title)
            nx.draw_networkx_nodes(
                G, em_fr_pos, node_size=15, node_color=color_list, edgecolors='white', linewidths=0.7)
            # nx.draw_networkx_labels(G, em_fr_pos, labels=labels, font_size=10)
            nx.draw_networkx_edges(G, em_fr_pos, width=0.2, alpha=0.2)
            plt.savefig(
                save_path
                + "/{}-EM-FR-{}s.png".format(graph_file.replace("/", "_").replace(".", "_"), em_fr_t)
            )
            plt.close()

            start = time.time()
            em_fr_pos = embedding_fr(G, vectors=vectors, te=te, wa=wa, we=we)
            end = time.time()
            em_fr_t = round(end - start, 3)

            plt.figure(figsize=(10, 10))
            plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
            fig_title = "{}-d{}-wl{}-ep{}-p{}-q{}-r{}-s{}-wa{}-we{}-te{}".format(
                "EM+FR", d, walklen, epochs, return_weight, neighbor_weight, attribute_weight, seed,
                wa, we, te
            )
            plt.title(fig_title)
            nx.draw_networkx_nodes(
                G, em_fr_pos, node_size=15, node_color=color_list, edgecolors='white', linewidths=0.7)
            # nx.draw_networkx_labels(G, em_fr_pos, labels=labels, font_size=10)
            nx.draw_networkx_edges(G, em_fr_pos, width=0.2, alpha=0.2)
            plt.savefig(
                save_path
                + "/{}-EM-FR-{}s.png".format(graph_file.replace("/", "_").replace(".", "_"), em_fr_t)
            )
            plt.close()
